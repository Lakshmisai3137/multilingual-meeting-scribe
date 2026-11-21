"""Offline inference shared by the API and the evaluation runner."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from array import array
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
from fastapi import HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .insights import Item, Segment, extract, sentences, summarize

MODELS = Path(__file__).resolve().parents[1] / "models"
Task = Literal["transcribe", "translate"]
RATE = 16000
MAX_SECONDS = 3600
MAX_BYTES = 128 * 1024 * 1024


class Meeting(BaseModel):
    task: Task
    language: str
    language_probability: float
    transcript_language: str
    summary_kind: Literal["generated", "verbatim", "no_speech", "unsupported_language"]
    summary: str
    action_items: list[str]
    items: list[Item]
    segments: list[Segment]


def decode(source, progress=lambda *args: None):
    samples = array("h")
    def append(frames):
        for frame in frames:
            if len(samples) + frame.samples > MAX_SECONDS * RATE:
                raise HTTPException(413, f"Audio exceeds {MAX_SECONDS // 60} minutes.")
            samples.frombytes(frame.to_ndarray().tobytes())
    try:
        with av.open(source, mode="r") as container:
            if not container.streams.audio:
                raise HTTPException(422, "Recording has no audio stream.")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
            for frame in container.decode(audio=0):
                progress("decoding", 0, 0)
                append(resampler.resample(frame))
            append(resampler.resample(None))
    except (av.error.FFmpegError, ValueError, EOFError, IndexError) as exc:
        raise HTTPException(422, "Could not decode the audio. Try another recording.") from exc
    if not samples:
        raise HTTPException(422, "Recording contains no audio samples.")
    return np.asarray(samples, dtype=np.float32) / np.float32(32768)


class Engine:
    def __init__(self):
        selected = os.getenv("SCRIBE_DEVICE", "auto").lower()
        if selected not in {"cpu", "cuda", "auto"}:
            raise ValueError("SCRIBE_DEVICE must be auto, cpu, or cuda")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if selected == "auto" else selected
        cuda = self.device == "cuda"
        whisper = MODELS / "faster-whisper-small"
        if not (whisper / "tokenizer.json").is_file():
            raise RuntimeError("Run python backend/download_models.py to prepare the local models.")
        self.whisper = WhisperModel(str(whisper), device=self.device, compute_type="float16" if cuda else "int8",
                                    local_files_only=True)
        bart = MODELS / "bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(bart, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(bart, local_files_only=True,
                                                       dtype=torch.float16 if cuda else torch.float32)
        self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer,
                                   framework="pt", device=0 if cuda else -1)

    def transcribe(self, audio, task, beam, progress):
        stage = "translating" if task == "translate" else "transcribing"
        duration = len(audio) / RATE
        progress(stage, 0, duration)
        peak = float(max(audio.max(), -audio.min()))
        if peak == 0:
            return [], "und", 0.0
        # Moderate gain on quiet recordings before VAD. Preserve normal speech levels.
        if peak < 0.08:
            audio = audio * min(16, 0.08 / peak)
        segments, info = self.whisper.transcribe(audio, task=task, beam_size=beam, vad_filter=True,
                                                 vad_parameters={"threshold": 0.35})
        if info.duration_after_vad <= 0:
            return [], "und", 0.0
        rows = []
        for segment in segments:
            if segment.text.strip():
                rows.append(Segment(start=segment.start, end=segment.end, text=segment.text.strip()))
            progress(stage, min(segment.end, duration), duration)
        return (rows, info.language, info.language_probability) if rows else ([], "und", 0.0)

    def process(self, source, task, progress):
        audio = decode(source, progress)
        segments, language, probability = self.transcribe(audio, task, 5, progress)
        del audio
        output = ("en" if task == "translate" else language) if segments else "und"
        items = []
        if not segments:
            summary, kind = "No speech detected.", "no_speech"
        elif output != "en":
            summary, kind = "Choose Translate to English for a summary and action items.", "unsupported_language"
        else:
            rows = list(sentences(segments))
            items = extract(rows)
            summary, kind = summarize(rows, self.summarizer, progress)
        return Meeting(task=task, language=language, language_probability=probability,
                       transcript_language=output, summary_kind=kind, summary=summary, items=items,
                       action_items=[item.text for item in items], segments=segments)
