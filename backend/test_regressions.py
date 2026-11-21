import asyncio
import io
from time import monotonic
from tempfile import SpooledTemporaryFile
import unittest
import wave
from unittest.mock import patch

import numpy as np
from starlette.datastructures import UploadFile
from starlette.requests import Request
from transformers import AutoConfig, AutoTokenizer, SummarizationPipeline
from . import engine
from .insights import Segment, encoder_length, extract, pack, sentences
from .main import Job, Jobs
from evaluate import chrf, distance, rouge, words


class RegressionTests(unittest.TestCase):
    def test_encoder_boundaries(self):
        path = engine.MODELS / "bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        pipeline = object.__new__(SummarizationPipeline)
        pipeline.tokenizer, pipeline.framework = tokenizer, "pt"
        pipeline.prefix = AutoConfig.from_pretrained(path, local_files_only=True).prefix
        def exact(text):
            return encoder_length(pipeline, text)
        for size in (1023, 1024, 1025):
            text = "localhost" + " hello" * (size-exact("localhost"))
            chunks = list(pack([text], exact, 1024))
            self.assertEqual(len(chunks), 1 if size <= 1024 else 2)
            self.assertEqual(" ".join(chunks), text)
            self.assertTrue(all(pipeline.preprocess(c)["input_ids"].shape[1] <= 1024 for c in chunks))

    def test_evidence_crosses_segments(self):
        rows = list(sentences([Segment(start=1,end=3,text="Maya, send the"),
                               Segment(start=4,end=6,text="slides by Friday. We decided to wait.")]))
        a,d = extract(rows)
        self.assertEqual((a.start,a.end,a.owner,a.due),(1,6,"Maya","Friday"))
        self.assertEqual((d.kind,d.start,d.end),("decision",4,6))

    def test_duration_checked_while_decoding(self):
        buffer = io.BytesIO()
        with wave.open(buffer,"wb") as f:
            f.setparams((1,2,16000,0,"NONE",""))
            f.writeframes(bytes(64000))
        buffer.seek(0)
        with patch.object(engine,"MAX_SECONDS",1), self.assertRaises(engine.HTTPException) as exc:
            engine.decode(buffer)
        self.assertEqual(exc.exception.status_code,413)

    def test_writable_multipart_spool_decodes_in_read_mode(self):
        with SpooledTemporaryFile(max_size=1024*1024) as stream:
            with wave.open(stream,"wb") as writer:
                writer.setparams((1,2,16000,0,"NONE",""))
                writer.writeframes(bytes(3200))
            stream.seek(0)
            self.assertEqual(engine.decode(stream).size,1600)

    def test_silence_never_reaches_whisper(self):
        self.assertEqual(engine.Engine.transcribe(None,np.zeros(16000), "transcribe",5,lambda *a: None),
                         ([], "und", 0.0))

    def test_metrics(self):
        self.assertEqual(words("हाँ, ठीक!"), ["हाँ","ठीक"])
        self.assertEqual(distance(["a","b"],["c","b"]),1)
        self.assertEqual(chrf("Résumé","Résumé"),1)
        self.assertEqual(chrf("abc","xyz"),0)
        self.assertEqual(rouge("a b c","a c")["rougeL"],.8)


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_queued_cancel_frees_slot_and_file(self):
        jobs = Jobs(None)
        j = Job(UploadFile(io.BytesIO(b"x"),filename="x.wav"),"transcribe")
        jobs.pending.append(j)
        jobs.admitted = 1
        await jobs.cancel(j)
        await jobs.cancel(j)
        self.assertTrue(j.upload.file.closed)
        self.assertEqual((jobs.admitted, len(jobs.pending),j.state),(0,0,"cancelled"))

    async def test_body_cap_with_and_without_content_length(self):
        boundary = b"test"
        body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n' + bytes(70000) + b'\r\n--test--\r\n'
        for known in (True, False):
            headers = [(b"content-type", b"multipart/form-data; boundary="+boundary)]
            if known:
                headers.append((b"content-length", str(len(body)).encode()))
            frames = iter([body[:1000], body[1000:]])
            async def receive():
                chunk = next(frames)
                return {"type":"http.request", "body":chunk, "more_body":len(chunk)==1000}
            request = Request({"type":"http", "headers":headers}, receive)
            jobs = Jobs(None)
            with patch("backend.main.MAX_BYTES", 50), self.assertRaises(engine.HTTPException) as exc:
                await jobs.submit(request)
            self.assertEqual(exc.exception.status_code,413)
            self.assertEqual(jobs.admitted,0)

    async def test_result_expiry(self):
        jobs = Jobs(None)
        j = Job(UploadFile(io.BytesIO(),filename="a.wav"),"transcribe")
        j.done.set()
        j.finished = monotonic()-901
        jobs.records[j.id] = j
        jobs.prune()
        self.assertNotIn(j.id,jobs.records)
        await j.upload.close()
