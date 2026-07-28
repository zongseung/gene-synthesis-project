"""학명 크로스워크 빌더.

이미지 라벨의 한국어 일반명(species_ko) / 612 생약명(herb_name)을 피벗 키인
**학명(scientific_name)** 으로 연결하고, 생약명·KP/KHP 수재 여부까지 채운다.
이름 매핑은 1:N(한 일반명 → 복수 기원종) 위험이 있어 confidence 로 표기하고,
모호/미해결은 다운스트림(safety_gate)에서 abstain 대상이 된다.

해상 전략(우선순위): manual seed > 외부 권위 소스(resolver) > 미해결.
외부 소스(NIFDS 공공API / OASIS 한약기원사전·이명)는 SpeciesResolver 로 교체 가능.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.knowledge.crosswalk \
      --inventory data/species_inventory.csv --seed data/crosswalk_seed.csv \
      --out data/crosswalk.parquet
"""
from __future__ import annotations
import argparse, csv, json, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class CrosswalkRow:
    dataset: str                         # 151 / 612 / both
    species_ko: str                      # 이미지 라벨 일반명 (조인 키)
    species_code: Optional[str] = None   # 151 코드
    herb_name: Optional[str] = None      # 생약명 (612 라벨 herb_name 또는 소스)
    scientific_name: Optional[str] = None  # 피벗 학명
    saengyak_std: Optional[str] = None   # 표준 생약명(라틴)
    kp_khp_id: Optional[str] = None      # 공정서 수재 식별자
    is_kp_accepted: Optional[bool] = None  # 한국 공정서 인정 기원종 여부
    synonyms: str = ""                   # 이명 (';'-join)
    confidence: str = "unresolved"       # resolved / ambiguous / unresolved
    source: str = ""                     # manual / nifds / oasis ...


class SpeciesResolver(Protocol):
    """일반명/생약명 → 학명 등 해상. 여러 후보면 confidence=ambiguous 로 신호."""
    name: str
    def resolve(self, species_ko: str, herb_name: Optional[str]) -> Optional[dict]:
        """매핑 dict(부분 채움 허용) 또는 None. 'confidence' 키로 ambiguous 신호 가능."""
        ...


class ManualSeedResolver:
    """수기/큐레이션 CSV. 컬럼: species_ko, scientific_name, saengyak_std,
    kp_khp_id, is_kp_accepted, synonyms, [confidence]. 최우선 신뢰."""
    name = "manual"

    def __init__(self, seed_csv: Optional[str]):
        self.table: dict[str, dict] = {}
        if seed_csv and os.path.exists(seed_csv):
            with open(seed_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    key = (r.get("species_ko") or "").strip()
                    if key:
                        self.table[key] = r

    def resolve(self, species_ko, herb_name):
        r = self.table.get(species_ko)
        if not r:
            return None
        out = {k: (r.get(k) or None) for k in
               ("scientific_name", "saengyak_std", "kp_khp_id", "synonyms", "herb_name")}
        if r.get("is_kp_accepted") not in (None, ""):
            out["is_kp_accepted"] = str(r["is_kp_accepted"]).strip().lower() in ("1", "true", "y", "yes")
        out["confidence"] = (r.get("confidence") or "resolved").strip()
        return {k: v for k, v in out.items() if v is not None}


class GbifResolver:
    """국명(vernacular) → 학명. GBIF Species API, 키 불필요. 식물계로 제한.
    한국어 vernacular 정확일치가 있으면 confidence=resolved, 후보 다수면 ambiguous."""
    name = "gbif"
    PLANTAE = 6

    def __init__(self, sleep: float = 0.15):
        self.sleep = sleep

    def _get(self, path, params):
        q = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"https://api.gbif.org/v1/{path}?{q}", timeout=20) as r:
            return json.load(r)

    def resolve(self, species_ko, herb_name):
        try:
            d = self._get("species/search", {
                "q": species_ko, "qField": "VERNACULAR", "rank": "SPECIES",
                "highertaxonKey": self.PLANTAE, "limit": 5})
        except Exception:
            return None
        finally:
            time.sleep(self.sleep)
        plants = [r for r in d.get("results", [])
                  if r.get("kingdom") == "Plantae" and r.get("canonicalName")]
        if not plants:
            return None

        def has_ko(r):
            return any(v.get("vernacularName") == species_ko and v.get("language") in ("kor", "ko")
                       for v in r.get("vernacularNames", []))
        exact = [r for r in plants if has_ko(r)]
        pool = exact or plants
        cand = pool[0]
        distinct = {r["canonicalName"] for r in pool}
        conf = "resolved" if (exact and len(distinct) == 1) else "ambiguous"
        return {"scientific_name": cand["canonicalName"], "confidence": conf}


class KpniResolver:
    """국명 → 학명. 국립수목원 국가표준식물목록 (data.go.kr 1400119/KpniService/scnmSearch).
    reqGnrlNm 은 부분일치 → plantGnrlNm 정확일치 + 정명(stpltScnmRltnCdNm) 으로 필터.
    자생식물 표준목록이라 자생/귀화종은 권위 있으나, 재배·외래종(강황 등)은 미수록."""
    name = "kpni"
    URL = "https://apis.data.go.kr/1400119/KpniService/scnmSearch"

    def __init__(self, service_key: str, sleep: float = 0.2):
        self.key = service_key
        self.sleep = sleep

    def resolve(self, species_ko, herb_name):
        params = urllib.parse.urlencode({
            "serviceKey": self.key, "pageNo": 1, "numOfRows": 50, "reqGnrlNm": species_ko})
        try:
            with urllib.request.urlopen(f"{self.URL}?{params}", timeout=20) as r:
                root = ET.fromstring(r.read())
        except Exception:
            return None
        finally:
            time.sleep(self.sleep)

        def field(it, tag):
            e = it.find(tag)
            return (e.text or "").strip() if e is not None and e.text else ""

        exact = [it for it in root.iter("item") if field(it, "plantGnrlNm") == species_ko]
        if not exact:
            return None
        accepted = [it for it in exact if field(it, "stpltScnmRltnCdNm") == "정명"]
        syn = sorted({field(it, "plantSpecsScnm") for it in exact
                      if field(it, "stpltScnmRltnCdNm") == "이명" and field(it, "plantSpecsScnm")})
        pool = accepted or exact
        names = sorted({field(it, "plantSpecsScnm") for it in pool if field(it, "plantSpecsScnm")})
        if not names:
            return None
        conf = "resolved" if (accepted and len(names) == 1) else "ambiguous"
        return {"scientific_name": names[0], "synonyms": ";".join(syn), "confidence": conf}


def _load_env_key(name: str) -> Optional[str]:
    env_path = "/home/user/gene-synthesis-project/.env"
    if os.environ.get(name):
        return os.environ[name]
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def load_inventory(path: str) -> list[CrosswalkRow]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(CrosswalkRow(
                dataset=r["dataset"], species_ko=r["species_ko"],
                species_code=(r.get("species_code") or None)))
    return rows


def build(inventory: list[CrosswalkRow], resolvers: list[SpeciesResolver]) -> list[CrosswalkRow]:
    """우선순위 resolver 체인으로 각 종을 채운다. 첫 해상에서 멈춘다."""
    for row in inventory:
        for rs in resolvers:
            got = rs.resolve(row.species_ko, row.herb_name)
            if not got:
                continue
            for k, v in got.items():
                if k == "confidence":
                    continue
                if hasattr(row, k) and getattr(row, k) in (None, "", False):
                    setattr(row, k, v)
            row.source = rs.name
            row.confidence = got.get("confidence", "resolved")
            if row.scientific_name:    # 학명까지 잡혔으면 종료
                break
    return inventory


def write_parquet(rows: list[CrosswalkRow], out: str):
    schema = pa.schema([
        ("dataset", pa.string()), ("species_ko", pa.string()), ("species_code", pa.string()),
        ("herb_name", pa.string()), ("scientific_name", pa.string()), ("saengyak_std", pa.string()),
        ("kp_khp_id", pa.string()), ("is_kp_accepted", pa.bool_()), ("synonyms", pa.string()),
        ("confidence", pa.string()), ("source", pa.string()),
    ])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows], schema=schema), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/species_inventory.csv")
    ap.add_argument("--seed", default="data/crosswalk_seed.csv")
    ap.add_argument("--out", default="data/crosswalk.parquet")
    args = ap.parse_args()

    inv = load_inventory(args.inventory)
    resolvers: list[SpeciesResolver] = [ManualSeedResolver(args.seed)]
    kpni_key = _load_env_key("DATA_GO_KR_KEY")
    if kpni_key:
        resolvers.append(KpniResolver(kpni_key))   # 자생종 권위 우선
    resolvers.append(GbifResolver())               # KPNI 미스 폴백
    # TODO: NIFDS(학명→생약명·KP/KHP) resolver — 해당 API 활용신청 후 추가
    print("resolvers:", [r.name for r in resolvers])
    rows = build(inv, resolvers)

    by_conf: dict[str, int] = {}
    for r in rows:
        by_conf[r.confidence] = by_conf.get(r.confidence, 0) + 1
    write_parquet(rows, args.out)
    print(f"크로스워크 {len(rows)}종 -> {args.out}")
    print(f"  해상도: {by_conf}")
    unresolved = [r.species_ko for r in rows if r.confidence == "unresolved"]
    print(f"  미해결 {len(unresolved)}종 (소스 연결 필요): {unresolved[:15]}{' ...' if len(unresolved)>15 else ''}")


if __name__ == "__main__":
    main()
