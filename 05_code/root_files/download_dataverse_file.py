"""Download a public Dataverse file with verified resumable byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_range(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: int,
) -> tuple[int, int]:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return start, expected

    temporary = part_path.with_suffix(part_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, timeout),
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(
                    f"bytes {start}-{end}/"
                ):
                    raise RuntimeError(
                        f"Unexpected range response {response.status_code}: {content_range!r}"
                    )
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"Range {start}-{end} has {temporary.stat().st_size} bytes; expected {expected}"
                )
            temporary.replace(part_path)
            return start, expected
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def download(
    url: str,
    output: Path,
    size: int,
    expected_md5: str,
    workers: int,
    chunk_size: int,
    retries: int,
    timeout: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.parent / f".{output.name}.parts"
    parts.mkdir(exist_ok=True)
    chunks = math.ceil(size / chunk_size)
    started = time.monotonic()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index in range(chunks):
            start = index * chunk_size
            end = min(size - 1, start + chunk_size - 1)
            part_path = parts / f"{index:05d}_{start}_{end}.part"
            futures.append(
                pool.submit(fetch_range, url, part_path, start, end, retries, timeout)
            )
        for future in as_completed(futures):
            _, length = future.result()
            completed += length
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"downloaded={completed / 1024**2:.1f} MiB "
                f"of={size / 1024**2:.1f} MiB "
                f"rate={completed / 1024**2 / elapsed:.2f} MiB/s",
                flush=True,
            )

    temporary_output = output.with_suffix(output.suffix + ".assembling")
    with temporary_output.open("wb") as destination:
        for index in range(chunks):
            start = index * chunk_size
            end = min(size - 1, start + chunk_size - 1)
            part_path = parts / f"{index:05d}_{start}_{end}.part"
            with part_path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(block)
    if temporary_output.stat().st_size != size:
        raise RuntimeError("Assembled file size does not match Dataverse metadata")
    actual_md5 = digest(temporary_output)
    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(f"MD5 mismatch: expected {expected_md5}, received {actual_md5}")
    temporary_output.replace(output)
    print(f"verified={output} md5={actual_md5}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    download(
        args.url,
        args.output,
        args.size,
        args.md5,
        args.workers,
        args.chunk_mib * 1024 * 1024,
        args.retries,
        args.timeout,
    )


if __name__ == "__main__":
    main()
