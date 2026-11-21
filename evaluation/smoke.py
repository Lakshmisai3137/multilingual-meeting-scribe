"""Exercise the live HTTP contract using local audio only."""
import argparse
import io
import json
from pathlib import Path
from time import monotonic, sleep
import wave

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
http = requests.Session()
http.trust_env = False


def wav(samples):
    stream = io.BytesIO()
    with wave.open(stream,"wb") as output:
        output.setparams((1,2,16000,0,"NONE",""))
        output.writeframes(samples.astype("<i2").tobytes())
    return stream.getvalue()


def smoke(base):
    def post(name, data, task="transcribe", route="/api/jobs"):
        return http.post(base+route,files={"file": (name,data)},data={"task":task},timeout=240)
    def wait(job):
        url=base+"/api/jobs/"+job["id"]
        deadline=monotonic()+300
        while monotonic()<deadline:
            state=http.get(url,timeout=10).json()
            if state["state"] in {"completed","cancelled","failed"}:
                return state
            sleep(.2)
        raise TimeoutError(job["id"])
    def result(name, data, task):
        accepted=post(name,data,task)
        assert accepted.status_code==202, accepted.text
        job=wait(accepted.json())
        assert job["state"]=="completed",job
        return http.get(base+"/api/jobs/"+job["id"]+"/result",timeout=10).json()

    print("Health",http.get(base+"/api/health",timeout=10).json(),flush=True)
    assert post("empty.wav",b"").status_code==422
    assert post("bad.txt",b"x").status_code==415
    assert post("bad.wav",b"x","invalid").status_code==422
    assert post("bad.wav",b"x",route="/api/process-meeting").status_code==422
    silence=wav(np.zeros(16000*8))
    silent=post("silent.wav",silence,route="/api/process-meeting").json()
    assert silent["summary_kind"]=="no_speech" and not silent["segments"], silent
    tick=wav(np.array([0, 900, 0]*100))
    brief=result("tick.wav",tick,"transcribe")
    assert brief["summary_kind"]=="no_speech" and not brief["segments"],brief
    manifest=json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    for lang in ["en","hi","es"]:
        path=ROOT/next(row["audio"] for row in manifest if row["language"]==lang)
        for task in ["transcribe","translate"]:
            data=result(path.name,path.read_bytes(),task)
            assert data["language"]==lang and data["segments"], (lang,data)
            assert data["transcript_language"]==("en" if task=="translate" else lang)
            if task=="transcribe" and lang!="en":
                assert data["summary_kind"]=="unsupported_language" and not data["items"]
            else:
                assert data["summary_kind"] in {"generated","verbatim"}
        print(lang,"both modes passed",flush=True)
    # The FLEURS WAVs use floating PCM. Decode and attenuate real speech to a peak of 0.5%.
    import av
    path=ROOT/manifest[0]["audio"]
    with av.open(str(path)) as container:
        resampler=av.AudioResampler(format="s16",layout="mono",rate=16000)
        parts=[out.to_ndarray().flatten() for frame in container.decode(audio=0) for out in resampler.resample(frame)]
        parts.extend(frame.to_ndarray().flatten() for frame in resampler.resample(None))
    samples=np.concatenate(parts).astype(np.float32)
    quiet=result("quiet.wav",wav(samples*164/np.abs(samples).max()),"transcribe")
    assert quiet["segments"] and quiet["language"]=="en",quiet
    print("Quiet speech preserved",flush=True)
    long_file=ROOT/"audio/ES2004a.wav"
    with long_file.open("rb") as f:
        first=post(long_file.name,f).json()
    url=base+"/api/jobs/"
    second=post("silence.wav",silence).json()
    third=post("silence.wav",silence).json()
    assert post("fourth.wav",silence).status_code==429
    assert http.delete(url+second["id"],timeout=10).json()["state"]=="cancelled"
    assert http.delete(url+first["id"],timeout=10).status_code==200
    assert wait(first)["state"]=="cancelled"
    assert wait(third)["state"]=="completed"
    assert http.get(url+first["id"]+"/result",timeout=10).status_code==409
    print("Queue limit, queued and active cancellation, and slot reuse passed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin",default="http://127.0.0.1:8000")
    smoke(parser.parse_args().origin.rstrip("/"))
