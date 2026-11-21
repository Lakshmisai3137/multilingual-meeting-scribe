"""Single-process offline API with bounded admission and cooperative cancellation."""
import asyncio
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .engine import Engine, MAX_BYTES, Meeting, Task

CAPACITY = 3
TTL = 15 * 60
MAX_RESULTS = 12
logger = logging.getLogger("uvicorn.error")


class Cancelled(Exception):
    pass


@dataclass
class Job:
    upload: UploadFile
    task: Task
    id: str = field(default_factory=lambda: uuid4().hex)
    state: str = "queued"
    progress: tuple = ("queued", 0, 0)
    stop: Event = field(default_factory=Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    result: Meeting | None = None
    error: str | None = None
    code: int = 500
    finished: float = 0

    def update(self, stage, current, total):
        if self.stop.is_set():
            raise Cancelled()
        self.progress = (stage, current, total)

    def view(self):
        stage, current, total = self.progress
        return dict(id=self.id, state=self.state, stage=stage, current=current, total=total,
                    filename=self.upload.filename, task=self.task, error=self.error)


class Jobs:
    def __init__(self, engine):
        self.engine = engine
        self.pending = deque()
        self.records = {}
        self.wake = asyncio.Event()
        self.admitted = 0
        self.closing = False

    def prune(self):
        completed = sorted((j for j in self.records.values() if j.done.is_set()), key=lambda j: j.finished)
        for i, job in enumerate(completed):
            if monotonic() - job.finished >= TTL or i < len(completed) - MAX_RESULTS:
                self.records.pop(job.id, None)

    async def submit(self, request):
        if self.closing or self.admitted >= CAPACITY:
            raise HTTPException(429, "Local engine is busy. Try again after a job finishes.")
        if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
            raise HTTPException(415, "Send a multipart audio upload.")
        self.admitted += 1
        form = None
        accepted = False
        try:
            limit = MAX_BYTES + 65536
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > limit:
                raise HTTPException(413, "Upload exceeds 128 MiB.")
            async def bounded_body():
                received = 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > limit:
                        raise HTTPException(413, "Upload exceeds 128 MiB.")
                    yield chunk

            async with asyncio.timeout(120):
                form = await MultiPartParser(request.headers, bounded_body(), max_files=1,
                                         max_fields=1, max_part_size=256).parse()
            upload = form.get("file")
            task = form.get("task", "transcribe")
            if not isinstance(upload, UploadFile) or task not in ("transcribe", "translate"):
                raise HTTPException(422, "Supply an audio file and a valid transcribe or translate task.")
            if Path(upload.filename or "").suffix.lower() not in {".wav", ".mp3", ".m4a"}:
                raise HTTPException(415, "Choose an MP3, WAV, or M4A recording.")
            if not upload.size:
                raise HTTPException(422, "This file is empty.")
            if upload.size > MAX_BYTES:
                raise HTTPException(413, "Upload exceeds 128 MiB.")
            job = Job(upload, task)
            self.records[job.id] = job
            self.pending.append(job)
            self.wake.set()
            accepted = True
            return job
        except TimeoutError as exc:
            raise HTTPException(408, "Upload timed out.") from exc
        except MultiPartException as exc:
            raise HTTPException(400, "Invalid multipart upload.") from exc
        finally:
            if not accepted:
                self.admitted -= 1
                if form is not None:
                    await form.close()

    async def finish(self, job, state):
        job.state = state
        await job.upload.close()
        job.finished = monotonic()
        job.done.set()
        self.admitted -= 1
        self.prune()

    async def cancel(self, job):
        if job.done.is_set() or job.state in {'cancelled', 'failed', 'completed'}:
            return
        job.stop.set()
        if job.state == "queued":
            self.pending.remove(job)
            await self.finish(job, "cancelled")
        else:
            job.state = "cancelling"

    async def run(self):
        while self.pending or not self.closing:
            if not self.pending:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), 30)
                except TimeoutError:
                    self.prune()
                continue
            job = self.pending.popleft()
            job.state = "running"
            try:
                result = await asyncio.to_thread(self.engine.process, job.upload.file, job.task, job.update)
                job.update("complete", 1, 1)
                job.result = result
                state = "completed"
            except Cancelled:
                state = "cancelled"
            except HTTPException as exc:
                state, job.code, job.error = "failed", exc.status_code, str(exc.detail)
            except RuntimeError as exc:
                logger.exception("Inference runtime failed")
                detail = str(exc).lower()
                if "out of memory" in detail:
                    state, job.code, job.error = "failed", 503, "Insufficient memory. Try a shorter recording or restart with SCRIBE_DEVICE=cpu."
                elif any(name in detail for name in ("cublas", "cudnn", "library cublas")):
                    state, job.code, job.error = "failed", 503, "CUDA libraries unavailable. Check the GPU setup or restart with SCRIBE_DEVICE=cpu."
                else:
                    state, job.error = "failed", "Inference failed. Check the local backend terminal."
            except Exception:
                logger.exception("Inference failed")
                state, job.error = "failed", "Processing failed. Check the local backend terminal."
            await self.finish(job, state)


@asynccontextmanager
async def lifespan(app):
    app.state.jobs = jobs = Jobs(Engine())
    worker = asyncio.create_task(jobs.run())
    try:
        yield
    finally:
        jobs.closing = True
        for job in list(jobs.records.values()):
            await jobs.cancel(job)
        jobs.wake.set()
        await worker


app = FastAPI(title="Meeting Scribe", lifespan=lifespan, docs_url=None, redoc_url=None)
default_origins = [f"http://{host}:{port}" for host in ("localhost","127.0.0.1") for port in (5173,4173)]
app.add_middleware(CORSMiddleware,
                   allow_origins=os.getenv("SCRIBE_ORIGINS", ",".join(default_origins)).split(","),
                   allow_methods=["GET","POST","DELETE"], allow_headers=["Content-Type"])


def find(job_id):
    jobs = app.state.jobs
    jobs.prune()
    if job_id not in jobs.records:
        raise HTTPException(404, "Job expired or engine restarted. Upload the recording again.")
    return jobs.records[job_id]


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": app.state.jobs.engine.device}


@app.post("/api/jobs", status_code=202)
async def submit(request: Request):
    return (await app.state.jobs.submit(request)).view()


@app.get("/api/jobs/{job_id}")
async def status(job_id: str):
    return find(job_id).view()


@app.delete("/api/jobs/{job_id}")
async def cancel(job_id: str):
    job = find(job_id)
    await app.state.jobs.cancel(job)
    return job.view()


@app.get("/api/jobs/{job_id}/result", response_model=Meeting)
async def result(job_id: str):
    job = find(job_id)
    if job.result is None:
        raise HTTPException(job.code if job.error else 409, job.error or f"Job is {job.state}.")
    return job.result


@app.post("/api/process-meeting", response_model=Meeting)
async def process_meeting(request: Request):
    """Compatibility endpoint using the same admission, validation and worker."""
    job = await app.state.jobs.submit(request)
    await job.done.wait()
    if job.result is None:
        raise HTTPException(job.code if job.error else 409, job.error or "Job cancelled.")
    return job.result
