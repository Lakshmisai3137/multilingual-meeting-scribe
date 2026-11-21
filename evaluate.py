"""Offline evaluation on local human references. Run from the project root."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
from statistics import mean, median
from threading import Event, Thread
from time import perf_counter
import unicodedata

from backend.engine import Engine, RATE, decode, torch
from backend.insights import Segment, extract, sentences, summarize
from backend.model_versions import REVISIONS
import psutil

ROOT = Path(__file__).resolve().parent


def words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in normalized).split()


def distance(a, b, lcs=False):
    prev = [0] * (len(b)+1) if lcs else list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        row = [0 if lcs else i]
        for j, y in enumerate(b, 1):
            if lcs:
                row.append(prev[j-1]+1 if x == y else max(prev[j], row[-1]))
            else:
                row.append(min(prev[j]+1, row[-1]+1, prev[j-1]+(x != y)))
        prev = row
    return prev[-1]


def rouge(reference, predicted):
    ref, hyp = words(reference), words(predicted)
    def ngrams(tokens, n):
        return Counter(zip(*(tokens[i:] for i in range(n))))
    scores = {}
    for n in (1, 2):
        a, b = ngrams(ref, n), ngrams(hyp, n)
        scores[f"rouge{n}"] = 2 * (a & b).total() / max(1, a.total()+b.total())
    scores["rougeL"] = 2 * distance(ref, hyp, lcs=True) / max(1, len(ref)+len(hyp))
    return scores


def chrf(reference, predicted):
    """Character n-gram F2, orders 1..6, no whitespace, effective orders."""
    ref, hyp = "".join(reference.split()), "".join(predicted.split())
    precisions, recalls = [], []
    for n in range(1, 7):
        a = Counter(ref[i:i+n] for i in range(len(ref)-n+1))
        b = Counter(hyp[i:i+n] for i in range(len(hyp)-n+1))
        if a and b:
            matches = (a & b).total()
            precisions.append(matches/b.total())
            recalls.append(matches/a.total())
    if not precisions:
        return 0.0
    precision, recall = mean(precisions), mean(recalls)
    return 5*precision*recall/(4*precision+recall) if precision and recall else 0.0


def extraction_scores(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for baseline in (True, False):
        tp = fp = fn = slots = 0
        for i, (text, kind, *fields) in enumerate(cases):
            items = extract([Segment(start=i, end=i+1, text=text)], baseline)
            match = bool(items and items[0].kind == kind)
            tp += match
            fp += bool(items and not match)
            fn += bool(kind and not match)
            slots += bool(match and [items[0].owner, items[0].due] == fields)
        output["keyword" if baseline else "structured"] = dict(
            cases=len(cases), tp=tp, fp=fp, fn=fn, precision=tp/max(1,tp+fp),
            recall=tp/max(1,tp+fn), f1=2*tp/max(1,2*tp+fp+fn), exact_including_fields=slots)
    return output


def timed_asr(engine, audio, task, beam):
    process = psutil.Process()
    stop = Event()
    peaks = [0, 0]
    def sample():
        peaks[0] = max(peaks[0], process.memory_info().rss)
        if engine.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            peaks[1] = max(peaks[1], total-free)
    def monitor():
        while not stop.wait(0.05):
            sample()
    sample()
    monitor_thread = Thread(target=monitor)
    monitor_thread.start()
    try:
        start = perf_counter()
        segments, detected, _ = engine.transcribe(audio, task, beam, lambda *args: None)
        elapsed = perf_counter() - start
        sample()
    finally:
        stop.set()
        monitor_thread.join()
    return segments, dict(seconds=elapsed, rtf=elapsed/(len(audio)/RATE), detected=detected,
                          peak_rss_mib=peaks[0]/2**20,
                          peak_device_used_mib=peaks[1]/2**20 if engine.device == "cuda" else None)


def evaluate(manifest, output):
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    engine = Engine()
    report = dict(device=engine.device, hardware=torch.cuda.get_device_name() if engine.device == "cuda"
                  else platform.processor(), torch=torch.__version__, revisions=REVISIONS, cases=[],
                  extraction=extraction_scores(ROOT/"evaluation/actions.json"))
    warmup = decode(str(manifest.parent/cases[0]["audio"]))
    for beam in (1,5):
        engine.transcribe(warmup, "transcribe", beam, lambda *args: None)
    for i, case in enumerate(cases):
        audio = decode(str(manifest.parent/case["audio"]))
        ref = words(case["transcript"])
        if not ref:
            raise ValueError(f"Empty reference for {case['id']}")
        row = dict(id=case["id"], language=case["language"], duration=len(audio)/RATE, beams={})
        for beam in ((1,5) if i%2 == 0 else (5,1)):
            segs, stats = timed_asr(engine, audio, "transcribe", beam)
            hyp = " ".join(s.text for s in segs)
            errors = distance(ref, words(hyp))
            row["beams"][str(beam)] = dict(**stats, errors=errors, reference_words=len(ref),
                                          wer=errors/len(ref), transcript=hyp)
            if beam == 5:
                source_segments = segs
        if case["language"] != "en" and case.get("english_reference"):
            translated, stats = timed_asr(engine, audio, "translate", 5)
            translation = " ".join(s.text for s in translated)
            row["translation"] = dict(text=translation, english_reference=case["english_reference"],
                                      chrf=chrf(case["english_reference"], translation), **stats)
        if case.get("summary"):
            if case["language"] != "en" and not case.get("english_reference"):
                raise ValueError("Supply an English reference transcript for English summaries.")
            gold = case["transcript"] if case["language"] == "en" else case["english_reference"]
            if case["language"] != "en":
                source_segments = translated
            gold_rows = list(sentences([Segment(start=0, end=row["duration"], text=gold)]))
            predicted_rows = list(sentences(source_segments))
            outputs = {"bart_reference": summarize(gold_rows, engine.summarizer)[0],
                       "bart_asr": summarize(predicted_rows, engine.summarizer)[0],
                       "lead3_asr": " ".join(s.text for s in predicted_rows[:3])}
            row["summaries"] = {k: dict(text=v, **rouge(case["summary"],v)) for k,v in outputs.items()}
            row["reference_summary"] = case["summary"]
            row["reference_decisions"] = case.get("final_decisions", [])
        report["cases"].append(row)
        print("Scored", case["id"], flush=True)
    by_language = defaultdict(list)
    for row in report["cases"]:
        by_language[row["language"]].append(row)
    report["asr"] = {}
    for lang, rows in by_language.items():
        report["asr"][lang] = {}
        for beam in ("1","5"):
            stats = [r["beams"][beam] for r in rows]
            report["asr"][lang][beam] = dict(
                n=len(rows), corpus_wer=sum(s["errors"] for s in stats)/sum(s["reference_words"] for s in stats),
                median_rtf=median(s["rtf"] for s in stats), peak_rss_mib=max(s["peak_rss_mib"] for s in stats),
                peak_device_used_mib=max((s["peak_device_used_mib"] or 0 for s in stats)) if engine.device=="cuda" else None)
    report["summary_scores"] = {method: {metric: mean(values) for metric in ("rouge1","rouge2","rougeL")
                                     if (values := [r["summaries"][method][metric] for r in report["cases"] if "summaries" in r])}
                                for method in ("bart_reference","bart_asr","lead3_asr")}
    report["translation_chrf"] = {
        lang: mean(r["translation"]["chrf"] for r in rows if "translation" in r)
        for lang, rows in by_language.items() if any("translation" in r for r in rows)}
    from evaluation.reversal import run as reversal
    report["scripted_reversal_summary"] = reversal(engine.summarizer)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"evaluation/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"evaluation/results.json")
    args = parser.parse_args()
    evaluate(args.manifest, args.output)
