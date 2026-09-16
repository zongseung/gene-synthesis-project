#!/usr/bin/env python3
"""
22개 염색체 VCF → 단일 파일 병합 (병렬 처리)

1) 각 염색체를 병렬로 읽어 haplotype 행렬 추출
2) 전체를 하나의 파일로 합침

출력 형식 선택:
  - "vcf"  : bcftools concat으로 합친 VCF.gz (기본)
  - "tsv"  : 0/1 haplotype 텍스트 행렬
  - "pkl"  : pandas DataFrame pickle (메모리 효율)
"""

import os
import sys
import time
import argparse
import pysam
from multiprocessing import Pool, cpu_count

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Allow direct execution: python src/preprocessing/merge_data.py
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (  # noqa: E402
    PER_CHROM_VCF_DIR as VCF_DIR,
    PER_CHROM_VCF_PATTERN as VCF_PATTERN,
)

CHROMOSOMES = list(range(1, 23))
N_WORKERS = min(22, cpu_count())
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")


def check_files():
    """모든 VCF 파일 존재 확인"""
    missing = []
    found = []
    for chrom in CHROMOSOMES:
        path = os.path.join(VCF_DIR, VCF_PATTERN.format(chrom=chrom))
        if os.path.exists(path):
            found.append(path)
        else:
            missing.append(f"chr{chrom}")
    if missing:
        print(f"WARNING: 누락된 염색체: {missing}")
    print(f"발견된 파일: {len(found)}/22")
    return found


# ──────────────────────────────────────────────
# 방법 1: pysam concat (VCF.gz 출력)
# ──────────────────────────────────────────────
def index_one(vcf_path):
    """단일 VCF에 tabix 인덱스 생성 (pysam 사용)"""
    tbi = vcf_path + ".tbi"
    if os.path.exists(tbi):
        return f"  [SKIP] {os.path.basename(vcf_path)} (인덱스 존재)"
    try:
        pysam.tabix_index(vcf_path, preset="vcf", force=True)
        return f"  [DONE] {os.path.basename(vcf_path)}"
    except Exception as e:
        return f"  [FAIL] {os.path.basename(vcf_path)}: {e}"


def _get_checkpoint_dir(output_path):
    """체크포인트 디렉토리 경로 반환"""
    out_dir = os.path.dirname(output_path) or "."
    return os.path.join(out_dir, ".vcf_merge_checkpoint")


def _compress_one_vcf(args):
    """단일 VCF를 임시 bgzf 파일로 복사 (병렬 워커용, 체크포인트 지원)"""
    vcf_path, tmp_path = args
    done_marker = tmp_path + ".done"

    # 이미 완료된 파일 건너뛰기
    if os.path.exists(done_marker) and os.path.exists(tmp_path):
        return f"  [SKIP] {os.path.basename(vcf_path)} (체크포인트 존재)"

    try:
        # 임시 파일에 먼저 쓰고 완료 후 마커 생성
        partial_path = tmp_path + ".partial"
        with pysam.VariantFile(vcf_path) as vin:
            with pysam.VariantFile(partial_path, "wz", header=vin.header) as vout:
                for record in vin:
                    vout.write(record)
        os.replace(partial_path, tmp_path)
        # 완료 마커 생성
        with open(done_marker, "w") as f:
            f.write("done")
        return f"  [DONE] {os.path.basename(vcf_path)}"
    except Exception as e:
        # 실패 시 불완전 파일 제거
        for p in (partial_path, tmp_path, done_marker):
            if os.path.exists(p):
                os.remove(p)
        return f"  [FAIL] {os.path.basename(vcf_path)}: {e}"


def merge_vcf_concat(vcf_files, output_path):
    """pysam으로 VCF 병합 — 읽기는 병렬, 쓰기는 순차 결합, 체크포인트 지원"""
    import shutil

    print(f"\n=== 1단계: 병렬 인덱싱 ({N_WORKERS} workers) ===")
    with Pool(N_WORKERS) as pool:
        results = pool.map(index_one, vcf_files)
    for r in results:
        print(r)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 체크포인트 디렉토리 (재실행 시 재사용)
    ckpt_dir = _get_checkpoint_dir(output_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    tmp_pairs = []
    pending_pairs = []
    for vcf_path in vcf_files:
        tmp_path = os.path.join(ckpt_dir, os.path.basename(vcf_path))
        tmp_pairs.append((vcf_path, tmp_path))
        done_marker = tmp_path + ".done"
        if os.path.exists(done_marker) and os.path.exists(tmp_path):
            print(f"  [SKIP] {os.path.basename(vcf_path)} (체크포인트 존재)")
        else:
            pending_pairs.append((vcf_path, tmp_path))

    print(f"\n=== 2단계: 병렬 읽기 ===")
    print(f"  체크포인트: {ckpt_dir}")
    print(f"  처리 대상: {len(pending_pairs)}/{len(tmp_pairs)}개")
    t0 = time.time()

    if pending_pairs:
        n_workers = min(N_WORKERS, len(pending_pairs))
        with Pool(n_workers) as pool:
            results = pool.map(_compress_one_vcf, pending_pairs)
        for r in results:
            print(r)
    else:
        print("  모든 파일 체크포인트 완료, 건너뜀")

    elapsed = time.time() - t0
    print(f"병렬 읽기 완료 ({elapsed:.0f}s)")

    # 모든 파일이 준비됐는지 확인
    failed = [pair[0] for pair in tmp_pairs if not os.path.exists(pair[1] + ".done")]
    if failed:
        print(f"ERROR: {len(failed)}개 염색체 처리 실패, 재실행하면 완료된 것은 건너뜁니다.")
        for f in failed:
            print(f"  - {os.path.basename(f)}")
        sys.exit(1)

    print(f"\n=== 3단계: 바이너리 결합 ===")
    # 헤더만 pysam으로 쓰고, 나머지는 raw bgzf 블록 복사 (레코드 파싱 없이 고속)
    tmp_output = output_path + ".tmp"
    print(f"출력: {output_path}")
    t0 = time.time()

    # 첫 번째 파일에서 헤더 추출 후 기록
    vin_header = pysam.VariantFile(vcf_files[0])
    vout = pysam.VariantFile(tmp_output, "wz", header=vin_header.header)
    vout.close()
    vin_header.close()

    # 헤더 뒤에 각 체크포인트 파일의 데이터 블록을 바이너리로 이어붙기
    # bgzf EOF 블록 (28 bytes)을 제거하면서 연결
    BGZF_EOF = b'\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00\x42\x43\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    BGZF_EOF_LEN = len(BGZF_EOF)

    with open(tmp_output, "r+b") as fout:
        # 헤더 파일의 EOF 블록 제거
        fout.seek(0, 2)  # 파일 끝으로
        fsize = fout.tell()
        fout.seek(fsize - BGZF_EOF_LEN)
        if fout.read(BGZF_EOF_LEN) == BGZF_EOF:
            fout.seek(fsize - BGZF_EOF_LEN)
            fout.truncate()

        for i, (_, tmp_path) in enumerate(tmp_pairs):
            print(f"  결합 중 [{i+1}/{len(tmp_pairs)}]: {os.path.basename(tmp_path)}", flush=True)
            with open(tmp_path, "rb") as fin:
                data = fin.read()
            # 마지막 파일이 아니면 EOF 블록 제거
            if i < len(tmp_pairs) - 1:
                if data[-BGZF_EOF_LEN:] == BGZF_EOF:
                    data = data[:-BGZF_EOF_LEN]
            fout.write(data)

    # 완료 후 최종 파일로 이동 (atomic)
    os.replace(tmp_output, output_path)

    elapsed = time.time() - t0
    print(f"결합 완료 ({elapsed:.0f}s)")

    print("\n=== 4단계: 출력 인덱싱 ===")
    pysam.tabix_index(output_path, preset="vcf", force=True)
    print("인덱싱 완료")

    # 체크포인트 정리
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("체크포인트 정리 완료")

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"\n결과: {output_path} ({size_gb:.1f} GB)")


def main():
    global N_WORKERS

    parser = argparse.ArgumentParser(description="22개 염색체 VCF 병합 (병렬)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 파일 경로")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"병렬 워커 수 (기본 {N_WORKERS})")
    args = parser.parse_args()

    N_WORKERS = args.workers

    if args.output is None:
        args.output = os.path.join(OUTPUT_DIR, "ALL.autosomes.phase3.genotypes.vcf.gz")

    print(f"출력: {args.output}")
    print(f"워커: {N_WORKERS}")
    print()

    vcf_files = check_files()
    if not vcf_files:
        print("ERROR: VCF 파일 없음")
        sys.exit(1)

    t_start = time.time()

    merge_vcf_concat(vcf_files, args.output)

    total = time.time() - t_start
    print(f"\n=== 전체 완료 ({total:.0f}s / {total/60:.1f}min) ===")


if __name__ == "__main__":
    main()
