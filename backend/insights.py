"""English sentence-level evidence extraction and token-budgeted meeting summaries."""
import re
from bisect import bisect_left
from typing import Literal

from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Item(BaseModel):
    kind: Literal["action", "decision"]
    text: str
    start: float
    end: float
    owner: str | None = None
    due: str | None = None


BOUNDARY = re.compile(r"(?<=[.!?。！？।])\s+|\n+")
KEYWORDS = re.compile(r"\b(?:will|need to|must|should|assigned to|deadline|todo|to-do|action item|decided)\b", re.I)
DECISION = re.compile(r"\b(?:decided|we agreed (?:to|on|that)|final decision|we (?:approved|rejected|rescinded|cancelled|canceled)|decision is)\b", re.I)
VERB = r"(?:send|share|review|prepare|update|submit|deliver|finish|draft|schedule|check|publish|contact|email|call|write|test|book|confirm|follow up|implement|investigate|organize|create|release|fix)"
ASSIGN = re.compile(rf"\b(?P<owner>[A-Z][\w’-]*)\s*(?:,\s*(?:please\s+)?|(?:\s+(?i:will|must|needs to|has to)|['’]ll)\s+)(?P<verb>(?i:{VERB}))\b")
COMMIT = re.compile(rf"\b(?:I|we)\s*(?:['’]ll|will|need to|must|can)\s+{VERB}\b|^(?:please\s+)?{VERB}\b|\b(?:action item|to-do|todo)\s*:", re.I)
DUE = re.compile(r"\b(?:by|before|due(?: on)?)\s+((?:next |this )?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month)|tomorrow|today|end of (?:day|week)|\d{4}-\d\d-\d\d|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b|\b(tomorrow|today)\b", re.I)


def sentences(segments):
    """Keep the full source time span even when ASR splits a sentence."""
    segments = [s for s in segments if s.text.strip()]
    text = " ".join(s.text.strip() for s in segments)
    ends, cursor = [], -1
    for s in segments:
        cursor += 1 + len(s.text.strip())
        ends.append(cursor)
    start = 0
    for match in [*BOUNDARY.finditer(text), None]:
        end = match.start() if match else len(text)
        if end > start:
            yield Segment(text=text[start:end], start=segments[bisect_left(ends, start+1)].start,
                          end=segments[bisect_left(ends, end)].end)
        if match:
            start = match.end()


def extract(rows, baseline=False):
    items = []
    for row in rows:
        text = row.text
        name = ASSIGN.search(text)
        if baseline:
            if len(text.split()) <= 4 or not KEYWORDS.search(text):
                continue
            kind = "decision" if re.search(r"\bdecided\b", text, re.I) else "action"
        elif DECISION.search(text):
            kind = "decision"
        elif name or COMMIT.search(text):
            kind = "action"
        else:
            continue
        owner = name["owner"] if name and kind == "action" and not baseline else None
        if owner and owner.lower() in {"i", "we", "you", "he", "she", "they", "team", "it"}:
            owner = None
        deadline = DUE.search(text) if kind == "action" and not baseline else None
        items.append(Item(**row.model_dump(), kind=kind, owner=owner,
                          due=next((v for v in deadline.groups() if v), None) if deadline else None))
    return items


def pack(texts, count, limit):
    """Greedily pack whole sentences. Bisect an oversized individual sentence."""
    def pieces(text):
        if count(text) <= limit:
            yield text
            return
        mid = len(text) // 2
        space = text.rfind(" ", mid // 2, mid + 1)
        cut = space + 1 if space >= 0 else mid
        if not 0 < cut < len(text):
            raise ValueError("Encoder budget cannot fit one character")
        yield from pieces(text[:cut].rstrip())
        yield from pieces(text[cut:].lstrip())

    current = ""
    for text in texts:
        for piece in pieces(text.strip()):
            combined = (current + " " + piece).strip()
            if current and count(combined) > limit:
                yield current
                current = piece
            else:
                current = combined
    if current:
        yield current


def encoder_length(pipeline, text):
    return len(pipeline.tokenizer((pipeline.prefix or "") + text, verbose=False)["input_ids"])


def summarize(rows, pipeline, progress=lambda *args: None):
    text = " ".join(r.text for r in rows)
    if len(text.split()) < 15:
        return text, "verbatim"
    limit = min(1024, pipeline.model.config.max_position_embeddings)
    def count(value):
        return encoder_length(pipeline, value)

    chunks = list(pack((r.text for r in rows), count, limit))
    latest = next((r for r in reversed(rows) if DECISION.search(r.text)), None) if len(chunks)>1 else None
    stage = "summarizing"
    while chunks:
        summaries = []
        for i, chunk in enumerate(chunks):
            progress(stage, i, len(chunks))
            maximum = min(142, max(4, count(chunk) // 2))
            result = pipeline(chunk, do_sample=False, truncation=False,
                              min_length=min(30, maximum // 2), max_length=maximum)
            summaries.append(result[0]["summary_text"].strip())
            progress(stage, i + 1, len(chunks))
        if len(summaries) == 1:
            summary = summaries[0]
            if latest and latest.text.casefold() not in summary.casefold():
                # Preserve the last explicitly worded decision with its source time.
                # Earlier decisions on unrelated topics still appear in the evidence list.
                seconds = int(latest.start)
                evidence = latest.text
                if len(evidence.split()) > 64:
                    clause = evidence[DECISION.search(evidence).start():].split()
                    evidence = " ".join(clause[:64]) + ("…" if len(clause) > 64 else "")
                summary += f"\nLatest recorded decision ({seconds//60:02}:{seconds%60:02}): {evidence}"
            return summary, "generated"
        chunks = list(pack(summaries, count, limit))
        stage = "synthesizing"
    return "", "no_speech"
