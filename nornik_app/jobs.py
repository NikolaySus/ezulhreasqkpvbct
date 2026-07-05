from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


class JobCancelled(Exception):
    pass


class JobFailed(Exception):
    pass


class JobTimedOut(Exception):
    pass


@dataclass
class _Job:
    process: mp.Process
    queue: mp.Queue


class JobManager:
    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._jobs: dict[str, _Job] = {}
        self._cancelled: set[str] = set()
        self._lock = self._context.RLock()

    def run(
        self,
        job_id: str,
        worker: Callable[..., dict[str, Any]],
        *args: Any,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        self.cancel(job_id)
        result_queue: mp.Queue = self._context.Queue(maxsize=1)
        process = self._context.Process(target=_run_worker, args=(worker, args, result_queue), daemon=True)
        with self._lock:
            self._cancelled.discard(job_id)
            self._jobs[job_id] = _Job(process=process, queue=result_queue)
        process.start()
        started_at = time.monotonic()

        try:
            while True:
                try:
                    message = result_queue.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    message = None

                if message is not None:
                    status, payload = message
                    if status == "ok":
                        return payload
                    raise JobFailed(str(payload))

                if not process.is_alive():
                    with self._lock:
                        cancelled = job_id in self._cancelled
                    if cancelled:
                        raise JobCancelled(job_id)
                    raise JobFailed(f"Worker exited without result, exitcode={process.exitcode}")

                if time.monotonic() - started_at > timeout:
                    self.cancel(job_id)
                    raise JobTimedOut(job_id)

                time.sleep(0.05)
        finally:
            self._cleanup(job_id, process, result_queue)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job:
                self._cancelled.add(job_id)
        if not job:
            return False
        if job.process.is_alive():
            job.process.terminate()
        job.process.join(timeout=2.0)
        if job.process.is_alive():
            job.process.kill()
            job.process.join(timeout=2.0)
        _close_queue(job.queue)
        return True

    def _cleanup(self, job_id: str, process: mp.Process, result_queue: mp.Queue) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current and current.process.pid == process.pid:
                self._jobs.pop(job_id, None)
            self._cancelled.discard(job_id)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        else:
            process.join(timeout=2.0)
        _close_queue(result_queue)


def _run_worker(worker: Callable[..., dict[str, Any]], args: tuple[Any, ...], result_queue: mp.Queue) -> None:
    try:
        result_queue.put(("ok", worker(*args)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def _close_queue(result_queue: mp.Queue) -> None:
    try:
        result_queue.close()
        result_queue.join_thread()
    except (OSError, ValueError):
        pass
