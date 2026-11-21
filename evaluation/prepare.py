"""Fetch the small public evaluation subset once. Inference and scoring remain offline."""
import csv
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data"
AMI = "https://groups.inf.ed.ac.uk/ami"
MEETING = "ES2004a"


def download(url, **kwargs):
    response = requests.get(url, timeout=(20, 120), **kwargs)
    response.raise_for_status()
    return response


def metadata(locale):
    text = download(f"{FLEURS}/{locale}/test.tsv").content.decode("utf-8")
    return {row[1]: row for row in csv.reader(text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)}


def fleurs():
    english = {r[0]: r[2] for r in metadata("en_us").values()}
    for locale in ("en_us", "hi_in", "es_419"):
        rows = metadata(locale)
        seen = set()
        with download(f"{FLEURS}/{locale}/audio/test.tar.gz", stream=True) as response:
            with tarfile.open(fileobj=response.raw, mode="r|gz") as bundle:
                for entry in bundle:
                    name = Path(entry.name).name
                    row = rows.get(name)
                    if not entry.isfile() or not row or row[0] in seen or row[0] not in english:
                        continue
                    seen.add(row[0])
                    target = AUDIO / f"{locale}-{name}"
                    target.write_bytes(bundle.extractfile(entry).read())
                    yield dict(id=target.stem, audio="audio/" + target.name, language=locale[:2],
                               transcript=row[2], english_reference=english[row[0]],
                               source=f"Google FLEURS test / {locale} / sentence {row[0]}")
                    if len(seen) == 3:
                        break
        if len(seen) != 3:
            raise RuntimeError(f"Incomplete {locale} subset")


def ami():
    target = AUDIO / f"{MEETING}.wav"
    with download(f"{AMI}/AMICorpusMirror/amicorpus/{MEETING}/audio/{MEETING}.Mix-Headset.wav", stream=True) as response:
        with target.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
    archive = download(f"{AMI}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip").content
    turns = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for speaker in "ABCD":
            base = f"{MEETING}.{speaker}"
            words = ET.fromstring(bundle.read(f"words/{base}.words.xml"))
            turns_xml = ET.fromstring(bundle.read(f"segments/{base}.segments.xml"))
            lookup = {int(w.attrib["{http://nite.sourceforge.net/}id"].split("words")[-1]): w.text or ""
                      for w in words if w.tag == "w"}
            for turn in turns_xml:
                ids = [int(i) for i in re.findall(r"words(\d+)\)", turn[0].attrib["href"])]
                text = " ".join(lookup[i] for i in range(ids[0], ids[-1]+1) if i in lookup)
                if text:
                    turns.append((float(turn.attrib["transcriber_start"]), re.sub(r"\s+([,.?!])", r"\1", text)))
        abstract = ET.fromstring(bundle.read(f"abstractive/{MEETING}.abssumm.xml"))
    return dict(id=MEETING, audio="audio/" + target.name, language="en",
                transcript=" ".join(text for _, text in sorted(turns)),
                summary=" ".join(s.text for s in abstract.findall("abstract/sentence")),
                final_decisions=[s.text for s in abstract.findall("decisions/sentence") if s.text and s.text.strip().upper() not in {"NA.", "NA", "N/A"}],
                source="AMI Meeting Corpus / ES2004a / manual annotations 1.6.2, mixed headset")


if __name__ == "__main__":
    AUDIO.mkdir(exist_ok=True)
    cases = [*fleurs(), ami()]
    path = ROOT / "manifest.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(cases)} reference recordings in {ROOT}")
