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
import subprocess
import time
import argparse
import numpy as np
import pandas as pd
import pickle
import pysam
from multiprocessing import Pool, cpu_count
from functools import partial

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

VCF_DIR = os.path.expanduser("~/GeneDiffusion")
VCF_PATTERN = "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
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


# ──────────────────────────────────────────────
# 방법 2: Haplotype 행렬 추출 (TSV / PKL 출력)
# ──────────────────────────────────────────────
def extract_haplotypes_one(chrom, biallelic_only=True, maf_threshold=0.0):
    """
    단일 염색체에서 haplotype 행렬 추출
    Returns: (chrom, positions_df, hap_matrix)
      - positions_df: DataFrame with CHROM, POS, REF, ALT
      - hap_matrix: numpy array (n_snps, n_haplotypes) of 0/1
    """
    from cyvcf2 import VCF

    vcf_path = os.path.join(VCF_DIR, VCF_PATTERN.format(chrom=chrom))
    if not os.path.exists(vcf_path):
        print(f"  [chr{chrom}] 파일 없음, 건너뜀", flush=True)
        return chrom, None, None

    t0 = time.time()
    print(f"  [chr{chrom}] 읽는 중...", flush=True)

    vcf = VCF(vcf_path)
    sample_names = vcf.samples
    n_samples = len(sample_names)

    positions = []   # (chrom, pos, ref, alt)
    hap_rows = []    # each row: 0/1 array of length 2*n_samples

    for variant in vcf:
        # biallelic SNP만
        if biallelic_only:
            if len(variant.ALT) != 1 or len(variant.REF) != 1 or len(variant.ALT[0]) != 1:
                continue

        # phased genotype 추출: gt_phases_array는 없으므로 직접 파싱
        gt_array = variant.genotype.array()  # shape: (n_samples, 3) → allele1, allele2, phased
        allele1 = gt_array[:, 0]  # 0 or 1
        allele2 = gt_array[:, 1]  # 0 or 1

        # missing (-1) 처리
        valid = (allele1 >= 0) & (allele2 >= 0)
        if valid.sum() == 0:
            continue

        # MAF 필터
        if maf_threshold > 0:
            af = (allele1[valid].sum() + allele2[valid].sum()) / (2.0 * valid.sum())
            maf = min(af, 1 - af)
            if maf < maf_threshold:
                continue

        # missing → 0으로 대체
        allele1[~valid] = 0
        allele2[~valid] = 0

        # 2개의 haplotype을 interleave: [a1_s1, a2_s1, a1_s2, a2_s2, ...]
        hap = np.empty(2 * n_samples, dtype=np.int8)
        hap[0::2] = allele1
        hap[1::2] = allele2
        hap_rows.append(hap)

        positions.append((f"chr{chrom}", variant.POS, variant.REF, variant.ALT[0]))

    elapsed = time.time() - t0

    if len(hap_rows) == 0:
        print(f"  [chr{chrom}] SNP 없음 ({elapsed:.0f}s)", flush=True)
        return chrom, None, None

    hap_matrix = np.stack(hap_rows)  # (n_snps, 2*n_samples)
    pos_df = pd.DataFrame(positions, columns=["CHROM", "POS", "REF", "ALT"])

    print(f"  [chr{chrom}] 완료: {len(hap_rows):,} SNPs × {2*n_samples} haplotypes ({elapsed:.0f}s)",
          flush=True)
    return chrom, pos_df, hap_matrix


def merge_haplotype_matrix(output_path, fmt="pkl", maf_threshold=0.0):
    """전체 염색체 haplotype 행렬 병합"""
    print(f"\n=== Haplotype 행렬 추출 ({N_WORKERS} workers) ===")

    extract_fn = partial(extract_haplotypes_one, maf_threshold=maf_threshold)
    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = pool.map(extract_fn, CHROMOSOMES)

    # 염색체 순서대로 합치기
    results.sort(key=lambda x: x[0])

    all_pos = []
    all_hap = []
    sample_names = None

    for chrom, pos_df, hap_matrix in results:
        if pos_df is not None:
            all_pos.append(pos_df)
            all_hap.append(hap_matrix)
            # 샘플 이름은 한번만 추출
            if sample_names is None:
                from cyvcf2 import VCF as _VCF
                vcf_path = os.path.join(VCF_DIR, VCF_PATTERN.format(chrom=chrom))
                sample_names = list(_VCF(vcf_path).samples)

    if not all_hap:
        print("ERROR: 추출된 데이터 없음")
        return

    positions_df = pd.concat(all_pos, ignore_index=True)
    hap_matrix = np.concatenate(all_hap, axis=0)  # (total_snps, 2*n_samples)

    elapsed = time.time() - t0
    print(f"\n병합 완료: {hap_matrix.shape[0]:,} SNPs × {hap_matrix.shape[1]:,} haplotypes ({elapsed:.0f}s)")

    # 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "tsv":
        print(f"TSV 저장 중: {output_path}")
        # 헤더: CHROM POS REF ALT sample1_h1 sample1_h2 sample2_h1 ...
        hap_cols = []
        for s in sample_names:
            hap_cols.extend([f"{s}_h1", f"{s}_h2"])
        hap_df = pd.DataFrame(hap_matrix, columns=hap_cols, dtype=np.int8)
        out_df = pd.concat([positions_df.reset_index(drop=True), hap_df], axis=1)
        out_df.to_csv(output_path, sep="\t", index=False)

    elif fmt == "pkl":
        print(f"Pickle 저장 중: {output_path}")
        data = {
            "positions": positions_df,
            "haplotypes": hap_matrix,        # np.int8, (n_snps, 2*n_samples)
            "sample_names": sample_names,
            "shape_info": f"{hap_matrix.shape[0]} SNPs × {hap_matrix.shape[1]} haplotypes"
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"저장 완료: {output_path} ({size_gb:.2f} GB)")
    print(f"  SNPs: {hap_matrix.shape[0]:,}")
    print(f"  Haplotypes: {hap_matrix.shape[1]:,} ({len(sample_names)} samples × 2)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    global N_WORKERS

    parser = argparse.ArgumentParser(description="22개 염색체 VCF 병합 (병렬)")
    parser.add_argument("--format", choices=["vcf", "tsv", "pkl"], default="vcf",
                        help="출력 형식: vcf (bcftools concat), tsv (텍스트 행렬), pkl (pickle)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 파일 경로")
    parser.add_argument("--maf", type=float, default=0.0,
                        help="MAF 필터 (tsv/pkl만, 기본 0=필터 없음)")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"병렬 워커 수 (기본 {N_WORKERS})")
    args = parser.parse_args()

    N_WORKERS = args.workers

    # 기본 출력 경로
    if args.output is None:
        ext_map = {"vcf": os.path.join(OUTPUT_DIR, "ALL.autosomes.phase3.genotypes.vcf.gz"),
                   "tsv": os.path.join(OUTPUT_DIR, "all_chromosomes_haplotypes.tsv"),
                   "pkl": os.path.join(OUTPUT_DIR, "all_chromosomes_haplotypes.pkl")}
        args.output = ext_map[args.format]

    print(f"형식: {args.format}")
    print(f"출력: {args.output}")
    print(f"워커: {N_WORKERS}")
    if args.maf > 0:
        print(f"MAF 필터: ≥ {args.maf}")
    print()

    vcf_files = check_files()
    if not vcf_files:
        print("ERROR: VCF 파일 없음")
        sys.exit(1)

    t_start = time.time()

    if args.format == "vcf":
        merge_vcf_concat(vcf_files, args.output)
    else:
        merge_haplotype_matrix(args.output, fmt=args.format, maf_threshold=args.maf)

    total = time.time() - t_start
    print(f"\n=== 전체 완료 ({total:.0f}s / {total/60:.1f}min) ===")


if __name__ == "__main__":
    main()
