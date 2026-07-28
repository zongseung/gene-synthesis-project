"""
Mediclassics 다중 서적 크롤러 — 권 목록은 HTML 스크래핑, content_total은 fetch-until-empty.

Per-book rate limit이 분리되어 있으므로 책별로 별도 프로세스로 병렬 실행 가능.

Usage:
  # 단일 책
  python mediclassics_multi_crawler.py --book-id 93 --output data/raw/mediclassics

  # 모든 Core 7 병렬 (백그라운드 실행)
  for ID in 8 56 69 86 93 182 291; do
    python mediclassics_multi_crawler.py --book-id $ID \
      --output data/raw/mediclassics > /tmp/crawl_$ID.log 2>&1 &
  done
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

BASE_URL = "https://mediclassics.kr"
AUTH_HEADER = {
    "Authorization": "5fe23edf9dec4c718e188073e46274bd",
    "Content-Type": "application/json",
}
DEFAULT_DELAY = 0.5
DEFAULT_CONCURRENCY = 2
DEFAULT_RATELIMIT_PAUSE = 60.0
MAX_RETRIES = 5
EMPTY_RUN_THRESHOLD = 3


async def fetch_book_metadata(client: httpx.AsyncClient, book_id: int) -> tuple[str, list[dict]]:
    """Book detail HTML에서 책 이름 + 권 목록 추출."""
    url = f"{BASE_URL}/books/{book_id}"
    for attempt in range(1, 4):
        try:
            resp = await client.get(url, timeout=120.0)
            resp.raise_for_status()
            html = resp.text
            break
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == 3:
                raise
            wait = 5.0 * attempt
            logging.warning(f"  metadata fetch attempt {attempt}/3 failed: {e!r}, retry in {wait}s")
            await asyncio.sleep(wait)

    title_match = re.search(r"var\s+book_nm_kor\s*=\s*'([^']+)'", html)
    book_name = title_match.group(1).encode("latin-1").decode("unicode_escape") if title_match else f"book_{book_id}"
    book_name = bytes(book_name, "utf-8").decode("utf-8", errors="ignore")

    vol_pattern = re.compile(rf'href="/books/{book_id}/volume/(\d+)">([^<|]+(?:\|[^<]+)?)')
    seen = set()
    volumes = []
    for vol_id_str, vol_nm in vol_pattern.findall(html):
        vid = int(vol_id_str)
        if vid in seen:
            continue
        seen.add(vid)
        volumes.append({
            "volume_id": vid,
            "volume_nm": vol_nm.strip(),
            "content_total": None,
        })

    volumes.sort(key=lambda v: v["volume_id"])
    return book_name, volumes


async def fetch_content(
    client: httpx.AsyncClient,
    log: logging.Logger,
    book_id: int,
    volume_id: int,
    seq: int,
    semaphore: asyncio.Semaphore,
    delay: float,
    pause: float,
) -> tuple[str, dict | None]:
    """Returns (status, payload). status in {ok, empty, fail}."""
    url = f"{BASE_URL}/books/{book_id}/volume/{volume_id}/content/{seq}"
    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                resp = await client.get(url, headers=AUTH_HEADER, timeout=30.0)
                if resp.status_code in (405, 429, 503):
                    log.warning(f"  vol={volume_id} seq={seq}: {resp.status_code} rate-limit, sleep {pause}s")
                    await asyncio.sleep(pause)
                    continue
                if resp.status_code == 404:
                    return "empty", None
                if resp.status_code == 200 and len(resp.content) == 0:
                    return "empty", None
                resp.raise_for_status()
                await asyncio.sleep(delay)
                return "ok", resp.json()
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
                wait = 2.0 ** attempt
                log.warning(f"  vol={volume_id} seq={seq}: {e!r} (retry {attempt}/{MAX_RETRIES}, wait {wait}s)")
                await asyncio.sleep(wait)
    log.error(f"  vol={volume_id} seq={seq}: FAILED after {MAX_RETRIES} retries")
    return "fail", None


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
    log: logging.Logger,
    book_id: int,
    volume: dict,
    output_dir: Path,
    resume: bool,
    delay: float,
    concurrency: int,
    pause: float,
) -> dict:
    vol_id = volume["volume_id"]
    vol_name = volume["volume_nm"]
    out_file = output_dir / f"vol_{vol_id:02d}.jsonl"

    existing_seqs: set[int] = set()
    max_existing_seq = 0
    if resume and out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing_seqs.add(rec["content_seq"])
                max_existing_seq = max(max_existing_seq, rec["content_seq"])
        log.info(f"  vol={vol_id} ({vol_name}): resume, {len(existing_seqs)} done, max_seq={max_existing_seq}")

    semaphore = asyncio.Semaphore(concurrency)
    fetched = 0
    failed = 0
    empty_run = 0
    seq = max_existing_seq + 1
    BATCH = 20

    while empty_run < EMPTY_RUN_THRESHOLD:
        batch_seqs = []
        while len(batch_seqs) < BATCH:
            if seq in existing_seqs:
                seq += 1
                continue
            batch_seqs.append(seq)
            seq += 1

        tasks = [fetch_content(client, log, book_id, vol_id, s, semaphore, delay, pause) for s in batch_seqs]
        results = await asyncio.gather(*tasks)

        with open(out_file, "a", encoding="utf-8") as f:
            for s, (status, raw) in zip(batch_seqs, results):
                if status == "empty":
                    empty_run += 1
                    continue
                if status == "fail":
                    failed += 1
                    empty_run = 0
                    continue
                empty_run = 0
                f.write(json.dumps(extract_record(raw), ensure_ascii=False) + "\n")
                fetched += 1

        log.info(f"  vol={vol_id}: seq={seq-1}, total_fetched={len(existing_seqs)+fetched}, fail={failed}, empty_run={empty_run}")

    total_in_file = len(existing_seqs) + fetched
    volume["content_total"] = total_in_file
    log.info(f"  vol={vol_id} ({vol_name}): DONE, content_total={total_in_file}, failed={failed}")
    return {"volume_id": vol_id, "name": vol_name, "fetched": fetched, "failed": failed, "content_total": total_in_file}


def write_manifest(output_dir: Path, book_id: int, book_name: str, volumes: list[dict], stats: list[dict]):
    vol_files = sorted(output_dir.glob("vol_*.jsonl"))
    file_hashes = {}
    total_records = ko_count = en_count = 0
    for vf in vol_files:
        sha = hashlib.sha256()
        lines = 0
        ko = en = 0
        with open(vf, "rb") as f:
            for line in f:
                sha.update(line)
                lines += 1
                rec = json.loads(line)
                if rec.get("trans_ko"):
                    ko += 1
                if rec.get("trans_en"):
                    en += 1
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
        "ko_coverage": f"{ko_count / total_records * 100:.1f}%" if total_records else "N/A",
        "en_coverage": f"{en_count / total_records * 100:.1f}%" if total_records else "N/A",
        "files": file_hashes,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Mediclassics 다중 서적 크롤러")
    parser.add_argument("--output", type=Path, required=True, help="root dir, will create book_{id}/ subdir")
    parser.add_argument("--book-id", type=int, required=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--pause", type=float, default=DEFAULT_RATELIMIT_PAUSE)
    args = parser.parse_args()

    book_dir = args.output / f"book_{args.book_id:03d}"
    book_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(f"book_{args.book_id}")
    log.info(f"=== Book {args.book_id} crawl start ===")
    log.info(f"  output={book_dir}, delay={args.delay}, concurrency={args.concurrency}, pause={args.pause}")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        book_name, volumes = await fetch_book_metadata(client, args.book_id)
        log.info(f"  book='{book_name}', volumes={len(volumes)}")
        for v in volumes[:5]:
            log.info(f"    vol_{v['volume_id']:02d}: {v['volume_nm']}")
        if len(volumes) > 5:
            log.info(f"    ... ({len(volumes) - 5} more)")

        stats = []
        for vol in volumes:
            stat = await crawl_volume(client, log, args.book_id, vol, book_dir, args.resume, args.delay, args.concurrency, args.pause)
            stats.append(stat)

        write_manifest(book_dir, args.book_id, book_name, volumes, stats)

    log.info(f"=== Book {args.book_id} DONE: {sum(s['fetched'] for s in stats)} fetched, {sum(s['failed'] for s in stats)} failed ===")


if __name__ == "__main__":
    asyncio.run(main())
