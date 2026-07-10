#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_ROOT = Path("training-artifacts/segmentation_dataset/all-1783214091-1/corrected-originals")


@dataclass
class Sample:
    timestamp: float
    pid: int
    sm: float
    mem: float
    fb_mb: float


def main() -> int:
    args = parse_args()
    image_path = args.image or first_image(DEFAULT_IMAGE_ROOT)
    if image_path is None:
        print("No test image found. Pass --image PATH.", file=sys.stderr)
        return 2
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    gpu_info = query_gpu_memory()
    worker_pids = docker_container_pids(args.container)
    if not worker_pids:
        print(f"No host PIDs found for container {args.container}. Is the worker running?", file=sys.stderr)
        return 2
    baseline = sample_gpu_baseline(worker_pids)

    print(f"API: {args.api_url}")
    print(f"Worker container: {args.container}")
    print(f"Initial worker PIDs: {', '.join(map(str, sorted(worker_pids)))}")
    print(f"Test image: {image_path}")
    print(f"GPU memory: total={gpu_info['total_mb']} MB, used={gpu_info['used_mb']} MB, free={gpu_info['free_mb']} MB")
    print(f"Baseline excluding worker: sm={baseline['sm']:.1f}% mem={baseline['mem']:.1f}% fb={baseline['fb_mb']:.0f} MB")

    stop_event = threading.Event()
    samples: list[Sample] = []
    monitor = threading.Thread(target=monitor_worker_gpu, args=(args.container, args.interval, stop_event, samples), daemon=True)
    monitor.start()

    started = time.monotonic()
    result: dict[str, object] = {}
    failure: BaseException | None = None
    try:
        job_id = submit_job(args.api_url, image_path)
        print(f"Job: {job_id}")
        result = wait_job(args.api_url, job_id, args.timeout)
    except BaseException as exc:
        failure = exc
    finally:
        stop_event.set()
        monitor.join(timeout=max(args.interval * 2, 2.0))

    elapsed = time.monotonic() - started
    print(f"Elapsed: {elapsed:.1f}s")
    if failure is not None:
        print(f"Job error: {failure}", file=sys.stderr)
    print(f"Job status: {result.get('status', 'unknown')}")
    if result.get("status") == "failed":
        print(json.dumps(result, indent=2, ensure_ascii=False))

    summary = summarize(samples)
    if summary["sample_count"] == 0:
        print("No worker GPU samples were captured. The job may have been too short or pmon did not see the process.", file=sys.stderr)
        return 1

    capacities = estimate_capacities(summary, gpu_info, baseline, args.safety, args.reserve_fb_mb, args.max_workers)
    print()
    print("Measured single-worker GPU usage")
    print(f"  samples: {summary['sample_count']}")
    if summary["sample_count"] < 10:
        print("  warning: few GPU samples captured; use a larger representative image for a safer estimate.")
    print(f"  sm:  avg={summary['avg_sm']:.1f}%  p95={summary['p95_sm']:.1f}%  peak={summary['peak_sm']:.1f}%")
    print(f"  mem: avg={summary['avg_mem']:.1f}%  p95={summary['p95_mem']:.1f}%  peak={summary['peak_mem']:.1f}%")
    print(f"  fb:  avg={summary['avg_fb_mb']:.0f} MB  peak={summary['peak_fb_mb']:.0f} MB")
    print()
    print("Capacity estimate by peak usage")
    print(f"  safety target: {args.safety * 100:.0f}%")
    print(f"  VRAM reserve: {args.reserve_fb_mb:.0f} MB")
    print_capacity("all GPU free for workers", capacities["ideal"])
    print_capacity("current background SM only", capacities["background_sm"])
    print_capacity("current background MEM only", capacities["background_mem"])
    print_capacity("current background FB only", capacities["background_fb"])
    print()
    print(f"Recommended workers now: {capacities['current_recommended']}")

    if capacities["current_recommended"] <= 1:
        print("Recommendation: keep one GPU worker.")
    else:
        print(f"Recommendation: up to {capacities['current_recommended']} GPU workers should be safe by these limits.")
    return 1 if failure is not None or result.get("status") == "failed" else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure GPU usage of one RQ worker and estimate safe worker count.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001", help="Backend API base URL.")
    parser.add_argument("--container", default="ezulhreasqkpvbct-worker", help="Docker container name of one GPU worker.")
    parser.add_argument("--image", type=Path, default=None, help="Image to submit for the benchmark.")
    parser.add_argument("--interval", type=float, default=1.0, help="nvidia-smi pmon sampling interval in seconds.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Job timeout in seconds.")
    parser.add_argument("--safety", type=float, default=0.8, help="Target fraction of SM/MEM utilization to allocate.")
    parser.add_argument("--reserve-fb-mb", type=float, default=2048.0, help="VRAM to leave unused.")
    parser.add_argument("--max-workers", type=int, default=16, help="Upper bound for printed estimate.")
    return parser.parse_args()


def first_image(root: Path) -> Path | None:
    if not root.exists():
        return None
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp"):
        images = sorted(root.glob(suffix))
        if images:
            return images[0]
    return None


def run_command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def docker_container_pids(container: str) -> set[int]:
    try:
        output = run_command(["docker", "top", container, "-eo", "pid"])
    except subprocess.CalledProcessError:
        return set()
    pids: set[int] = set()
    for line in output.splitlines()[1:]:
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def query_gpu_memory() -> dict[str, float]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    total, used, free = [float(value.strip()) for value in output.split(",")[:3]]
    return {"total_mb": total, "used_mb": used, "free_mb": free}


def monitor_worker_gpu(container: str, interval: float, stop_event: threading.Event, samples: list[Sample]) -> None:
    while not stop_event.is_set():
        pids = docker_container_pids(container)
        if pids:
            samples.extend(sample_pmon(pids))
        stop_event.wait(interval)


def sample_pmon(target_pids: set[int]) -> list[Sample]:
    try:
        output = run_command(["nvidia-smi", "pmon", "-s", "um", "-c", "1"])
    except subprocess.CalledProcessError:
        return []
    now = time.monotonic()
    samples: list[Sample] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid not in target_pids:
            continue
        samples.append(
            Sample(
                timestamp=now,
                pid=pid,
                sm=parse_metric(parts[3]),
                mem=parse_metric(parts[4]),
                fb_mb=parse_metric(parts[9]),
            )
        )
    return samples


def sample_gpu_baseline(excluded_pids: set[int]) -> dict[str, float]:
    rows = sample_all_pmon_rows()
    sm = 0.0
    mem = 0.0
    fb_mb = 0.0
    for row in rows:
        if int(row["pid"]) in excluded_pids:
            continue
        sm += float(row["sm"])
        mem += float(row["mem"])
        fb_mb += float(row["fb_mb"])
    return {"sm": min(sm, 100.0), "mem": min(mem, 100.0), "fb_mb": fb_mb}


def sample_all_pmon_rows() -> list[dict[str, float | int]]:
    try:
        output = run_command(["nvidia-smi", "pmon", "-s", "um", "-c", "1"])
    except subprocess.CalledProcessError:
        return []
    rows: list[dict[str, float | int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        rows.append({"pid": pid, "sm": parse_metric(parts[3]), "mem": parse_metric(parts[4]), "fb_mb": parse_metric(parts[9])})
    return rows


def parse_metric(value: str) -> float:
    return 0.0 if value == "-" else float(value)


def submit_job(api_url: str, image_path: Path) -> str:
    content_type = content_type_for(image_path)
    boundary = f"----gpu-worker-benchmark-{int(time.time() * 1000)}"
    data = image_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/segment",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with open_url_no_proxy(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["job_id"])


def wait_job(api_url: str, job_id: str, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_payload: dict[str, object] = {}
    while time.monotonic() < deadline:
        with open_url_no_proxy(f"{api_url.rstrip('/')}/api/jobs/{job_id}", timeout=30) as response:
            last_payload = json.loads(response.read().decode("utf-8"))
        status = last_payload.get("status")
        if status in {"finished", "failed"}:
            return last_payload
        time.sleep(1.0)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout:.0f}s; last status: {last_payload}")


def open_url_no_proxy(request: urllib.request.Request | str, timeout: float):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def summarize(samples: list[Sample]) -> dict[str, float]:
    sm_values = [sample.sm for sample in samples]
    mem_values = [sample.mem for sample in samples]
    fb_values = [sample.fb_mb for sample in samples]
    return {
        "sample_count": len(samples),
        "avg_sm": average(sm_values),
        "p95_sm": percentile(sm_values, 95),
        "peak_sm": max(sm_values),
        "avg_mem": average(mem_values),
        "p95_mem": percentile(mem_values, 95),
        "peak_mem": max(mem_values),
        "avg_fb_mb": average(fb_values),
        "peak_fb_mb": max(fb_values),
    }


def estimate_capacities(
    summary: dict[str, float],
    gpu_info: dict[str, float],
    baseline: dict[str, float],
    safety: float,
    reserve_fb_mb: float,
    max_workers: int,
) -> dict[str, dict[str, int] | int]:
    ideal = estimate_capacity(summary, gpu_info, safety, reserve_fb_mb, max_workers, background_sm=0.0, background_mem=0.0, background_fb_mb=0.0)
    background_sm = estimate_capacity(
        summary,
        gpu_info,
        safety,
        reserve_fb_mb,
        max_workers,
        background_sm=baseline["sm"],
        background_mem=0.0,
        background_fb_mb=0.0,
    )
    background_mem = estimate_capacity(
        summary,
        gpu_info,
        safety,
        reserve_fb_mb,
        max_workers,
        background_sm=0.0,
        background_mem=baseline["mem"],
        background_fb_mb=0.0,
    )
    background_fb = estimate_capacity(
        summary,
        gpu_info,
        safety,
        reserve_fb_mb,
        max_workers,
        background_sm=0.0,
        background_mem=0.0,
        background_fb_mb=baseline["fb_mb"],
    )
    current_recommended = min(
        background_sm["by_sm"],
        background_mem["by_mem"],
        background_fb["by_fb"],
        max_workers,
    )
    return {
        "ideal": ideal,
        "background_sm": background_sm,
        "background_mem": background_mem,
        "background_fb": background_fb,
        "current_recommended": max(1, current_recommended),
    }


def estimate_capacity(
    summary: dict[str, float],
    gpu_info: dict[str, float],
    safety: float,
    reserve_fb_mb: float,
    max_workers: int,
    *,
    background_sm: float,
    background_mem: float,
    background_fb_mb: float,
) -> dict[str, int]:
    usable_sm = max(100.0 * safety - background_sm, 0.0)
    usable_mem = max(100.0 * safety - background_mem, 0.0)
    usable_fb = max(gpu_info["total_mb"] - reserve_fb_mb - background_fb_mb, 0.0)
    by_sm = bounded_capacity(usable_sm / summary["peak_sm"] if summary["peak_sm"] > 0 else max_workers, max_workers)
    by_mem = bounded_capacity(usable_mem / summary["peak_mem"] if summary["peak_mem"] > 0 else max_workers, max_workers)
    by_fb = bounded_capacity(usable_fb / summary["peak_fb_mb"] if summary["peak_fb_mb"] > 0 else max_workers, max_workers)
    recommended = max(1, min(by_sm, by_mem, by_fb, max_workers))
    return {"by_sm": by_sm, "by_mem": by_mem, "by_fb": by_fb, "recommended": recommended}


def print_capacity(label: str, capacity: dict[str, int]) -> None:
    print(
        f"  {label}: recommended={capacity['recommended']} "
        f"(by_sm={capacity['by_sm']}, by_mem={capacity['by_mem']}, by_fb={capacity['by_fb']})"
    )


def bounded_capacity(value: float, max_workers: int) -> int:
    if not math.isfinite(value):
        return max_workers
    return max(1, min(max_workers, int(math.floor(value))))


def average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(math.ceil((percent / 100.0) * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


if __name__ == "__main__":
    raise SystemExit(main())
