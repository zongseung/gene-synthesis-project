"""
Mediclassics 통합 오케스트레이터.

설계:
- ProcessPoolExecutor: N = min(cpu_count, num_books) — 책당 1 process
- 각 worker process 내부: asyncio + httpx (concurrency=2 per book, per-book rate limit 회피)
- Bulletproof retry: 에러 유형별 분리, exponential backoff, 무한 재시도(rate limit) vs 제한 재시도(network)
- 통합 manifest, 통합 로그, 통합 진행 상황
- Resume 모드 기본 활성

Usage:
  python mediclassics_orchestrator.py \
    --output data/raw/mediclassics_unified \
    --books 8,56,69,86,93,182,291

  # 진행 모니터:
  tail -f data/raw/mediclassics_unified/orchestrator.log
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ver4 r2 §5.5 재현성: set_global_seed 호출 (main 진입부에서 실행).
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402

BASE_URL = "https://mediclassics.kr"
AUTH_HEADER = {
    "Authorization": "5fe23edf9dec4c718e188073e46274bd",
    "Content-Type": "application/json",
}

DEFAULT_DELAY = 0.5
DEFAULT_CONCURRENCY = 2
DEFAULT_RATELIMIT_PAUSE = 60.0
DEFAULT_RATELIMIT_MAX_RETRIES = 30
DEFAULT_NETWORK_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_CAP = 60.0
EMPTY_RUN_THRESHOLD = 3
META_TIMEOUT = 120.0
CONTENT_TIMEOUT = 30.0
META_MAX_RETRIES = 3


def setup_worker_logger(book_id: int, log_dir: Path) -> logging.Logger:
    log = logging.getLogger(f"book_{book_id}")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log
    fh = logging.FileHandler(log_dir / f"book_{book_id:03d}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    return log


async def fetch_book_metadata(client: httpx.AsyncClient, log: logging.Logger, book_id: int) -> tuple[str, list[dict]]:
    url = f"{BASE_URL}/books/{book_id}"
    last_exc: Exception | None = None
    for attempt in range(1, META_MAX_RETRIES + 1):
        try:
            resp = await client.get(url, timeout=META_TIMEOUT)
            resp.raise_for_status()
            html = resp.text
            break
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
            last_exc = e
            wait = min(DEFAULT_BACKOFF_BASE ** attempt, DEFAULT_BACKOFF_CAP)
            log.warning(f"meta fetch attempt {attempt}/{META_MAX_RETRIES} failed: {e!r}; retry in {wait:.1f}s")
            await asyncio.sleep(wait)
    else:
        raise RuntimeError(f"meta fetch failed after {META_MAX_RETRIES} retries") from last_exc

    m = re.search(r"var\s+book_nm_kor\s*=\s*'([^']+)'", html)
    book_name = m.group(1).encode("latin-1").decode("unicode_escape") if m else f"book_{book_id}"

    vol_pattern = re.compile(rf'href="/books/{book_id}/volume/(\d+)">([^<|]+(?:\|[^<]+)?)')
    seen, volumes = set(), []
    for vol_id_str, vol_nm in vol_pattern.findall(html):
        vid = int(vol_id_str)
        if vid in seen:
            continue
        seen.add(vid)
        volumes.append({"volume_id": vid, "volume_nm": vol_nm.strip(), "content_total": None})
    volumes.sort(key=lambda v: v["volume_id"])
    return book_name, volumes


class FetchResult:
    OK = "ok"
    EMPTY = "empty"
    SKIP = "skip"
    FAIL = "fail"


async def fetch_content(
    client: httpx.AsyncClient,
    log: logging.Logger,
    book_id: int,
    volume_id: int,
    seq: int,
    semaphore: asyncio.Semaphore,
    delay: float,
    pause: float,
    rl_max_retries: int,
    net_max_retries: int,
) -> tuple[str, dict | None]:
    """
    Returns (status, payload).
      - "ok"   : payload dict
      - "empty": end-of-volume sentinel (HTTP 200 + empty body, or 404)
      - "skip" : 5xx repeated, give up after max retries
      - "fail" : unrecoverable error
    """
    url = f"{BASE_URL}/books/{book_id}/volume/{volume_id}/content/{seq}"
    rl_attempts = 0
    net_attempts = 0
    while True:
        async with semaphore:
            try:
                resp = await client.get(url, headers=AUTH_HEADER, timeout=CONTENT_TIMEOUT)
                status = resp.status_code

                if status == 404:
                    return FetchResult.EMPTY, None
                if status == 200 and len(resp.content) == 0:
                    return FetchResult.EMPTY, None

                if status in (405, 429, 503):
                    rl_attempts += 1
                    if rl_attempts > rl_max_retries:
                        log.error(f"vol={volume_id} seq={seq}: rate-limit FAIL after {rl_attempts} retries")
                        return FetchResult.FAIL, None
                    wait = pause * min(rl_attempts, 5)
                    log.warning(f"vol={volume_id} seq={seq}: HTTP {status} rate-limit (attempt {rl_attempts}/{rl_max_retries}), sleep {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                if 500 <= status < 600:
                    net_attempts += 1
                    if net_attempts > net_max_retries:
                        log.error(f"vol={volume_id} seq={seq}: server {status} SKIP after {net_attempts} retries")
                        return FetchResult.SKIP, None
                    wait = min(DEFAULT_BACKOFF_BASE ** net_attempts, DEFAULT_BACKOFF_CAP)
                    log.warning(f"vol={volume_id} seq={seq}: HTTP {status} (attempt {net_attempts}/{net_max_retries}), backoff {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                try:
                    payload = resp.json()
                except json.JSONDecodeError as je:
                    net_attempts += 1
                    if net_attempts > net_max_retries:
                        log.error(f"vol={volume_id} seq={seq}: JSON decode FAIL: {je!r}")
                        return FetchResult.SKIP, None
                    wait = min(DEFAULT_BACKOFF_BASE ** net_attempts, DEFAULT_BACKOFF_CAP)
                    log.warning(f"vol={volume_id} seq={seq}: JSON decode error, retry in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                await asyncio.sleep(delay)
                return FetchResult.OK, payload

            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
                net_attempts += 1
                if net_attempts > net_max_retries:
                    log.error(f"vol={volume_id} seq={seq}: network SKIP after {net_attempts} retries: {e!r}")
                    return FetchResult.SKIP, None
                wait = min(DEFAULT_BACKOFF_BASE ** net_attempts, DEFAULT_BACKOFF_CAP)
                log.warning(f"vol={volume_id} seq={seq}: {e!r} (attempt {net_attempts}/{net_max_retries}), backoff {wait:.1f}s")
                await asyncio.sleep(wait)


def extract_record(raw: dict) -> dict:
    return {
        "book_id": raw.get("book_id"),
        "volume_id": raw.get("volume_id"),
        "content_seq": raw.get("content_seq"),
        "content_level": (raw.get("content_level", "") or "") + (raw.get("content_level_depth", "") or ""),
        "up_path_nm": raw.get("up_path_nm"),
        "original": raw.get("original"),
        "trans_ko": raw.get("trans_2"),
        "trans_en": raw.get("trans_1"),
        "annotation": raw.get("annotation") if raw.get("annotation") else None,
        "index_num": raw.get("index_num"),
    }


async def crawl_volume(
    client, log, book_id, volume, output_dir, delay, concurrency, pause,
    rl_max_retries, net_max_retries,
) -> dict:
    vol_id = volume["volume_id"]
    vol_name = volume["volume_nm"]
    out_file = output_dir / f"vol_{vol_id:02d}.jsonl"

    existing_seqs: set[int] = set()
    max_existing_seq = 0
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_seqs.add(rec["content_seq"])
                    max_existing_seq = max(max_existing_seq, rec["content_seq"])
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info(f"vol={vol_id} ({vol_name}): resume {len(existing_seqs)} done, max_seq={max_existing_seq}")

    semaphore = asyncio.Semaphore(concurrency)
    fetched = failed = skipped = 0
    empty_run = 0
    seq = max_existing_seq + 1
    BATCH = 20
    out_handle = open(out_file, "a", encoding="utf-8")
    try:
        while empty_run < EMPTY_RUN_THRESHOLD:
            batch_seqs = []
            while len(batch_seqs) < BATCH:
                if seq in existing_seqs:
                    seq += 1
                    continue
                batch_seqs.append(seq)
                seq += 1

            tasks = [
                fetch_content(client, log, book_id, vol_id, s, semaphore, delay, pause, rl_max_retries, net_max_retries)
                for s in batch_seqs
            ]
            results = await asyncio.gather(*tasks)

            for s, (status, raw) in zip(batch_seqs, results):
                if status == FetchResult.EMPTY:
                    empty_run += 1
                elif status == FetchResult.SKIP:
                    skipped += 1
                    empty_run = 0
                elif status == FetchResult.FAIL:
                    failed += 1
                    empty_run = 0
                else:
                    empty_run = 0
                    out_handle.write(json.dumps(extract_record(raw), ensure_ascii=False) + "\n")
                    fetched += 1
            out_handle.flush()
            log.info(f"vol={vol_id}: seq={seq-1}, fetched_total={len(existing_seqs)+fetched}, fail={failed}, skip={skipped}, empty_run={empty_run}")
    finally:
        out_handle.close()

    total = len(existing_seqs) + fetched
    volume["content_total"] = total
    log.info(f"vol={vol_id} ({vol_name}): DONE total={total}, failed={failed}, skipped={skipped}")
    return {"volume_id": vol_id, "name": vol_name, "fetched": fetched, "failed": failed, "skipped": skipped, "content_total": total}


def write_book_manifest(output_dir: Path, book_id: int, book_name: str, volumes: list[dict], stats: list[dict]):
    vol_files = sorted(output_dir.glob("vol_*.jsonl"))
    file_hashes = {}
    total_records = ko_count = en_count = 0
    for vf in vol_files:
        sha = hashlib.sha256()
        lines = ko = en = 0
        with open(vf, "rb") as f:
            for line in f:
                sha.update(line)
                lines += 1
                try:
                    rec = json.loads(line)
                    if rec.get("trans_ko"):
                        ko += 1
                    if rec.get("trans_en"):
                        en += 1
                except json.JSONDecodeError:
                    pass
        total_records += lines
        ko_count += ko
        en_count += en
        file_hashes[vf.name] = {"sha256": sha.hexdigest(), "records": lines, "ko": ko, "en": en}

    manifest = {
        "source": "mediclassics.kr",
        "book_id": book_id,
        "book_name": book_name,
        "crawl_date": datetime.now(timezone.utc).isoformat(),
        "api_base": BASE_URL,
        "volumes_meta": volumes,
        "stats": stats,
        "actual_records": total_records,
        "records_with_ko": ko_count,
        "records_with_en": en_count,
        "ko_coverage_pct": round(ko_count / total_records * 100, 2) if total_records else 0,
        "en_coverage_pct": round(en_count / total_records * 100, 2) if total_records else 0,
        "files": file_hashes,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


async def crawl_book_async(book_id: int, root_dir: Path, log_dir: Path, delay: float, concurrency: int, pause: float, rl_max_retries: int, net_max_retries: int) -> dict:
    log = setup_worker_logger(book_id, log_dir)
    book_dir = root_dir / f"book_{book_id:03d}"
    book_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== Book {book_id} START ===")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        book_name, volumes = await fetch_book_metadata(client, log, book_id)
        log.info(f"book='{book_name}', volumes={len(volumes)}")

        stats = []
        for vol in volumes:
            try:
                stat = await crawl_volume(client, log, book_id, vol, book_dir, delay, concurrency, pause, rl_max_retries, net_max_retries)
                stats.append(stat)
            except Exception as e:
                log.exception(f"vol={vol['volume_id']} crashed: {e!r}")
                stats.append({"volume_id": vol["volume_id"], "error": repr(e)})

        write_book_manifest(book_dir, book_id, book_name, volumes, stats)

    summary = {
        "book_id": book_id,
        "book_name": book_name,
        "fetched": sum(s.get("fetched", 0) for s in stats),
        "failed": sum(s.get("failed", 0) for s in stats),
        "skipped": sum(s.get("skipped", 0) for s in stats),
        "volumes": len(volumes),
    }
    log.info(f"=== Book {book_id} DONE: {summary} ===")
    return summary


def crawl_book_worker(book_id: int, root_dir_str: str, log_dir_str: str, delay: float, concurrency: int, pause: float, rl_max_retries: int, net_max_retries: int) -> dict:
    """Process worker entry point."""
    try:
        return asyncio.run(crawl_book_async(book_id, Path(root_dir_str), Path(log_dir_str), delay, concurrency, pause, rl_max_retries, net_max_retries))
    except KeyboardInterrupt:
        return {"book_id": book_id, "error": "interrupted"}
    except Exception as e:
        return {"book_id": book_id, "error": repr(e)}


def write_unified_manifest(root_dir: Path, summaries: list[dict]):
    total_records = ko_count = en_count = 0
    for book_summary in summaries:
        bid = book_summary.get("book_id")
        if not bid:
            continue
        manifest_path = root_dir / f"book_{bid:03d}" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            total_records += m.get("actual_records", 0)
            ko_count += m.get("records_with_ko", 0)
            en_count += m.get("records_with_en", 0)
            book_summary["actual_records"] = m.get("actual_records", 0)
            book_summary["ko_coverage_pct"] = m.get("ko_coverage_pct", 0)
            book_summary["en_coverage_pct"] = m.get("en_coverage_pct", 0)

    unified = {
        "source": "mediclassics.kr",
        "crawl_date": datetime.now(timezone.utc).isoformat(),
        "books": summaries,
        "total_records": total_records,
        "total_ko": ko_count,
        "total_en": en_count,
        "ko_coverage_pct": round(ko_count / total_records * 100, 2) if total_records else 0,
        "en_coverage_pct": round(en_count / total_records * 100, 2) if total_records else 0,
    }
    with open(root_dir / "unified_manifest.json", "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--books", type=str, required=True, help="comma-separated book_ids")
    parser.add_argument("--workers", type=int, default=None, help="default = min(cpu, num_books)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--pause", type=float, default=DEFAULT_RATELIMIT_PAUSE)
    parser.add_argument("--rl-max-retries", type=int, default=DEFAULT_RATELIMIT_MAX_RETRIES)
    parser.add_argument("--net-max-retries", type=int, default=DEFAULT_NETWORK_MAX_RETRIES)
    parser.add_argument("--seed", type=int, default=42, help="전역 재현성 seed (ver4 r2 §5.5)")
    args = parser.parse_args()

    # ver4 r2 §5.5: 재현성 시드 설정 (random/numpy/torch + cuBLAS 결정성)
    set_global_seed(args.seed)

    books = [int(x) for x in args.books.split(",")]
    n_workers = args.workers if args.workers else min(os.cpu_count() or 1, len(books))

    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(exist_ok=True)

    root_log = logging.getLogger("orchestrator")
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(args.output / "orchestrator.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    root_log.addHandler(fh)
    root_log.addHandler(sh)

    root_log.info(f"=== ORCHESTRATOR START ===")
    root_log.info(f"  cpu={os.cpu_count()}, books={books}, workers={n_workers}")
    root_log.info(f"  delay={args.delay}, concurrency={args.concurrency}, pause={args.pause}")
    root_log.info(f"  retries: rl={args.rl_max_retries}, net={args.net_max_retries}")
    root_log.info(f"  output={args.output}, per-book log={log_dir}")

    t0 = time.time()
    summaries = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(
                crawl_book_worker, bid, str(args.output), str(log_dir),
                args.delay, args.concurrency, args.pause,
                args.rl_max_retries, args.net_max_retries,
            ): bid for bid in books
        }
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                summary = fut.result()
                summaries.append(summary)
                root_log.info(f"book_{bid} returned: {summary}")
            except Exception as e:
                root_log.exception(f"book_{bid} crashed in pool: {e!r}")
                summaries.append({"book_id": bid, "error": repr(e)})

    elapsed = time.time() - t0
    root_log.info(f"=== ALL DONE in {elapsed:.0f}s ===")
    write_unified_manifest(args.output, summaries)
    root_log.info(f"unified manifest: {args.output / 'unified_manifest.json'}")


if __name__ == "__main__":
    main()
