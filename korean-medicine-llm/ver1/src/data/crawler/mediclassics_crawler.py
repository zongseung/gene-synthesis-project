"""
Mediclassics.kr 동의보감 3중 병렬 코퍼스 수집기 (한문 + 국역 + 영역)

API endpoints (verified 2026-04-16):
  - Volume list: GET /api/books/{book_id}/volumes/
  - Content:     GET /books/{book_id}/volume/{vol_id}/content/{seq}

Usage:
  python mediclassics_crawler.py --output data/raw/mediclassics --book-id 8
  python mediclassics_crawler.py --output data/raw/mediclassics --book-id 8 --resume
  python mediclassics_crawler.py --output data/raw/mediclassics --book-id 8 --volumes 1,2,3
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://mediclassics.kr"
AUTH_HEADER = {
    "Authorization": "5fe23edf9dec4c718e188073e46274bd",
    "Content-Type": "application/json",
}
REQUEST_DELAY = 0.5
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
CONCURRENT_LIMIT = 2
RATELIMIT_PAUSE = 60.0


async def fetch_volumes(client: httpx.AsyncClient, book_id: int) -> list[dict]:
    url = f"{BASE_URL}/api/books/{book_id}/volumes/"
    resp = await client.get(url, headers=AUTH_HEADER)
    resp.raise_for_status()
    data = resp.json()
    volumes = data["DATA"]
    log.info(f"Book {book_id}: {len(volumes)} volumes, total {sum(v['content_total'] for v in volumes)} records")
    return volumes


async def fetch_content(
    client: httpx.AsyncClient,
    book_id: int,
    volume_id: int,
    seq: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    url = f"{BASE_URL}/books/{book_id}/volume/{volume_id}/content/{seq}"
    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                resp = await client.get(url, headers=AUTH_HEADER, timeout=30.0)
                if resp.status_code == 404:
                    log.warning(f"  vol={volume_id} seq={seq}: 404 (skip)")
                    return None
                if resp.status_code in (405, 429, 503):
                    wait = RATELIMIT_PAUSE
                    log.warning(f"  vol={volume_id} seq={seq}: {resp.status_code} rate-limit, sleep {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                await asyncio.sleep(REQUEST_DELAY)
                return resp.json()
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
                wait = RETRY_BACKOFF ** attempt
                log.warning(f"  vol={volume_id} seq={seq}: {e} (retry {attempt}/{MAX_RETRIES}, wait {wait:.1f}s)")
                await asyncio.sleep(wait)
    log.error(f"  vol={volume_id} seq={seq}: FAILED after {MAX_RETRIES} retries")
    return None


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
    client: httpx.AsyncClient,
    book_id: int,
    volume: dict,
    output_dir: Path,
    resume: bool,
) -> dict:
    vol_id = volume["volume_id"]
    vol_name = volume["volume_nm"]
    total = volume["content_total"]
    out_file = output_dir / f"vol_{vol_id:02d}.jsonl"

    existing_seqs: set[int] = set()
    if resume and out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing_seqs.add(rec["content_seq"])
        log.info(f"  vol={vol_id} ({vol_name}): resume, {len(existing_seqs)}/{total} already done")

    remaining = [s for s in range(1, total + 1) if s not in existing_seqs]
    if not remaining:
        log.info(f"  vol={vol_id} ({vol_name}): already complete ({total} records)")
        return {"volume_id": vol_id, "name": vol_name, "fetched": 0, "failed": 0, "total": total}

    log.info(f"  vol={vol_id} ({vol_name}): fetching {len(remaining)}/{total} records")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    fetched = 0
    failed = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        tasks = [fetch_content(client, book_id, vol_id, seq, semaphore) for seq in batch]
        results = await asyncio.gather(*tasks)

        with open(out_file, "a", encoding="utf-8") as f:
            for raw in results:
                if raw is None:
                    failed += 1
                    continue
                record = extract_record(raw)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                fetched += 1

        done = len(existing_seqs) + fetched + failed
        log.info(f"  vol={vol_id}: {done}/{total} ({fetched} ok, {failed} fail)")

    return {"volume_id": vol_id, "name": vol_name, "fetched": fetched, "failed": failed, "total": total}


def write_manifest(output_dir: Path, book_id: int, volumes: list[dict], stats: list[dict]):
    manifest = {
        "source": "mediclassics.kr",
        "book_id": book_id,
        "crawl_date": datetime.now(timezone.utc).isoformat(),
        "api_base": BASE_URL,
        "volumes": len(volumes),
        "total_records": sum(v["content_total"] for v in volumes),
        "stats": stats,
    }

    vol_files = sorted(output_dir.glob("vol_*.jsonl"))
    file_hashes = {}
    total_records_actual = 0
    ko_count = 0
    en_count = 0
    for vf in vol_files:
        sha = hashlib.sha256()
        lines = 0
        with open(vf, "rb") as f:
            for line in f:
                sha.update(line)
                lines += 1
                rec = json.loads(line)
                if rec.get("trans_ko"):
                    ko_count += 1
                if rec.get("trans_en"):
                    en_count += 1
        total_records_actual += lines
        file_hashes[vf.name] = {"sha256": sha.hexdigest(), "records": lines}

    manifest["actual_records"] = total_records_actual
    manifest["records_with_ko"] = ko_count
    manifest["records_with_en"] = en_count
    manifest["ko_coverage"] = f"{ko_count / total_records_actual * 100:.1f}%" if total_records_actual else "N/A"
    manifest["en_coverage"] = f"{en_count / total_records_actual * 100:.1f}%" if total_records_actual else "N/A"
    manifest["files"] = file_hashes

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"Manifest: {manifest_path}")
    log.info(f"  Total: {total_records_actual} records, KO: {ko_count} ({manifest['ko_coverage']}), EN: {en_count} ({manifest['en_coverage']})")


async def main():
    parser = argparse.ArgumentParser(description="Mediclassics 3중 병렬 코퍼스 크롤러")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--book-id", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--volumes", type=str, default=None, help="comma-sep volume IDs (e.g. 1,2,3)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--concurrency", type=int, default=CONCURRENT_LIMIT)
    args = parser.parse_args()

    globals()["REQUEST_DELAY"] = args.delay
    globals()["CONCURRENT_LIMIT"] = args.concurrency
    log.info(f"Config: delay={args.delay}s, concurrency={args.concurrency}, ratelimit_pause={RATELIMIT_PAUSE}s")

    args.output.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        volumes = await fetch_volumes(client, args.book_id)

        if args.volumes:
            target_ids = {int(x) for x in args.volumes.split(",")}
            volumes = [v for v in volumes if v["volume_id"] in target_ids]
            log.info(f"Filtered to {len(volumes)} volumes: {[v['volume_nm'] for v in volumes]}")

        stats = []
        for vol in volumes:
            stat = await crawl_volume(client, args.book_id, vol, args.output, args.resume)
            stats.append(stat)

        write_manifest(args.output, args.book_id, volumes, stats)

    total_fetched = sum(s["fetched"] for s in stats)
    total_failed = sum(s["failed"] for s in stats)
    log.info(f"Done. fetched={total_fetched}, failed={total_failed}")


if __name__ == "__main__":
    asyncio.run(main())
