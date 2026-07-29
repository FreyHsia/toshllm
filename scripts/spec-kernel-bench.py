#!/usr/bin/env python3
"""Benchmark speculative decoding through llama-server with repeatable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


DEFAULT_PROMPT = """Continue the Python code below. Output code only. Implement every missing read method following exactly the repetitive style already established, then add exhaustive pytest tests.

class ByteReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read_u8(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return value

    def read_u16_le(self) -> int:
        value = int.from_bytes(self.data[self.pos:self.pos + 2], "little")
        self.pos += 2
        return value

    # Implement read_u16_be, read_u32_le, read_u32_be, read_u64_le,
    # read_u64_be, read_i16_le, read_i32_le, read_f32_le, read_bytes,
    # skip, remaining, seek, and the tests.
"""


def request_json(port: int, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_for_health(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("ar", "mtp", "dflash"), required=True)
    parser.add_argument("--draft")
    parser.add_argument("--label", default="run")
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--draft-max", type=int)
    parser.add_argument("--content-preview", type=int, default=0)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()

    if args.mode == "dflash" and not args.draft:
        parser.error("--draft is required for dflash")

    env = os.environ.copy()
    env.update({"GGML_METAL_CONCURRENCY_DISABLE": "1", "TOSH_FA_AMD": "1"})
    for item in args.env:
        key, value = item.split("=", 1)
        env[key] = value

    command = [
        args.binary,
        "-m", args.model,
        "-ngl", "99",
        "-c", "4096",
        "-t", "6",
        "-fa", "1",
        "--no-mmap",
        "--parallel", "1",
        "--cache-ram", "256",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    if args.mode == "mtp":
        command += ["--spec-type", "draft-mtp"]
    elif args.mode == "dflash":
        command += [
            "-md", args.draft,
            "--spec-type", "draft-dflash",
            "-ngld", "99",
            "-ctkd", "q8_0",
            "-ctvd", "q8_0",
        ]
    if args.draft_max is not None:
        command += ["--spec-draft-n-max", str(args.draft_max)]

    stderr_lines: list[str] = []
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def drain_stderr() -> None:
        assert process.stderr is not None
        stderr_lines.extend(process.stderr)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    payload = {
        "prompt": DEFAULT_PROMPT * args.prompt_repeat,
        "n_predict": args.tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": args.ignore_eos,
    }

    failure: BaseException | None = None
    try:
        wait_for_health(args.port, process, 240)
        if args.warmup_tokens > 0:
            warmup = dict(payload)
            warmup["n_predict"] = args.warmup_tokens
            request_json(args.port, warmup, 600)

        results = []
        for rep in range(1, args.reps + 1):
            response = request_json(args.port, payload, 600)
            timings = response["timings"]
            content = response.get("content", "")
            drafted = int(timings.get("draft_n", 0) or 0)
            accepted = int(timings.get("draft_n_accepted", 0) or 0)
            row = {
                "rep": rep,
                "tg": float(timings["predicted_per_second"]),
                "pp": float(timings["prompt_per_second"]),
                "predicted_n": int(timings["predicted_n"]),
                "draft_n": drafted,
                "accepted": accepted,
                "acceptance": accepted / drafted if drafted else None,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
            results.append(row)
            print(json.dumps({"label": args.label, **row}), flush=True)
            if args.content_preview:
                print(json.dumps({
                    "label": args.label,
                    "rep": rep,
                    "content_length": len(content),
                    "content_preview": content[:args.content_preview],
                }), flush=True)

        speeds = [row["tg"] for row in results]
        summary = {
            "label": args.label,
            "mode": args.mode,
            "count": len(speeds),
            "low": min(speeds),
            "high": max(speeds),
            "mean": statistics.fmean(speeds),
            "median": statistics.median(speeds),
            "stdev": statistics.pstdev(speeds),
            "acceptance_mean": statistics.fmean(
                row["acceptance"] for row in results if row["acceptance"] is not None
            ) if any(row["acceptance"] is not None for row in results) else None,
            "hashes": sorted({row["sha256"] for row in results}),
        }
        print("SUMMARY " + json.dumps(summary), flush=True)
    except BaseException as error:
        failure = error
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        stderr_thread.join(timeout=2)

    selected_lines = [
        line for line in stderr_lines
        if "moe-profile:" in line or "draft acceptance" in line
    ]
    if failure is not None:
        selected_lines.extend(stderr_lines[-80:])
    for line in selected_lines:
        print("SERVER " + line.rstrip())
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
