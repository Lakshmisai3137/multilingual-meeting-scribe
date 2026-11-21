"""Prepare pinned model files online, or verify the installed copies offline."""
import argparse
import hashlib
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import hf_hub_download, snapshot_download
if __package__:
    from .model_versions import REVISIONS
else:
    from model_versions import REVISIONS

MODELS = Path(__file__).resolve().parents[1] / "models"
DIGESTS = {
    "Systran/faster-whisper-small": {
        "config.json": "e5047537059bd8f182d9ca64c470201585015187",
        "model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
        "tokenizer.json": "7818adb6de9fa3064d3ff81226fdd675be1f6344",
        "vocabulary.txt": "c9074644d9d1205686f16d411564729461324b75"},
    "facebook/bart-large-cnn": {
        "config.json": "2d297163111d6483909070dc6de8d34236c72636",
        "generation_config.json": "f729590ec05c715e3b922996b3a7b5c1c73fb1f7",
        "model.safetensors": "40041830399afb5348525ef8354b007ecec4286fdf3524f7e6b54377e17096cb",
        "tokenizer.json": "ad0bcbeb288f0d1373d88e0762e66357f55b8311",
        "merges.txt": "226b0752cac7789c48f0cb3ec53eda48b7be36cc",
        "vocab.json": "5606f48548d99a9829d10a96cd364b816b02cd21"},
}


def matches(path, expected):
    if not path.is_file():
        return False
    h = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        h.update(f"blob {path.stat().st_size}".encode() + bytes(1))
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest() == expected


def prepare(verify_only=False):
    for repo, files in DIGESTS.items():
        target = MODELS / repo.split("/")[-1]
        if not verify_only:
            snapshot_download(repo_id=repo, revision=REVISIONS[repo], local_dir=target, allow_patterns=list(files))
        for name, expected in files.items():
            path = target / name
            if matches(path, expected):
                continue
            if not verify_only:
                hf_hub_download(repo_id=repo, filename=name, revision=REVISIONS[repo],
                                local_dir=target, force_download=True)
                if matches(path, expected):
                    continue
            raise RuntimeError(f"Model integrity check failed for {path}. Run the downloader online to repair it.")
        print(f"Verified {repo} at {REVISIONS[repo]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    prepare(parser.parse_args().verify_only)
