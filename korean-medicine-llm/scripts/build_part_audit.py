"""부위 라벨 감사 시트 생성 — 블라인드 판정용 단일 HTML.

claudedocs/vlm_plan 의 M2(답변에 부위 포함) 결정 재료.

라벨이 「사진의 주피사체」와 맞는지를 사람이 직접 재는 도구다. **라벨을 먼저 보여주면
안 된다** — 보고 나면 동의하는 쪽으로 기울어 노이즈율이 실제보다 낮게 나온다.
그래서 판정할 때는 사진만 보이고, 라벨은 전부 끝난 뒤에 대조한다.

판정 선택지에 「여러 기관이 비슷하게 보임」을 둔 이유: 실물을 보면 줄기 사진에 잎이
함께 찍혀 있다. 그건 라벨 오류가 아니라 **배타적이지 않은 것**이라 따로 세야 한다.

산출: outputs/part_audit/part_audit.html  (이미지 내장, 오프라인 동작)
사용:
  PYTHONPATH=src .venv/bin/python scripts/build_part_audit.py --per_part 20
  브라우저로 outputs/part_audit/part_audit.html 열기
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import random
import sys

sys.path.insert(0, "src")

PART_KO = {"leaf": "잎", "stem": "줄기", "flower": "꽃", "fruit": "열매",
           "root": "뿌리", "whole": "전초", "group": "군락"}
# 판정 선택지. 라벨 어휘 + 배타성 검사용 2개.
CHOICES = [("leaf", "잎"), ("stem", "줄기"), ("flower", "꽃"), ("fruit", "열매"),
           ("root", "뿌리"), ("whole", "전초(한 개체 전체)"), ("group", "군락(여러 개체)"),
           ("multi", "여러 기관이 비슷하게 보임"), ("unclear", "판단 불가")]


def sample(rows_path: str, per_part: int, seed: int):
    """학습에 실제로 쓰는 행에서 부위별 per_part 장. 종이 겹치지 않게 고른다."""
    by_part = collections.defaultdict(list)
    seen = set()
    with open(rows_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("task") != "id":
                continue
            k = (r["part"], r["image"])
            if k not in seen:
                seen.add(k)
                by_part[r["part"]].append(r)
    rng = random.Random(seed)
    out = []
    for part, rows in sorted(by_part.items()):
        # 종당 최대 2장 — 한 종이 표본을 지배하면 노이즈율이 그 종 사진 품질이 된다
        per_sp = collections.defaultdict(list)
        for r in rows:
            per_sp[r["species_ko"]].append(r)
        pool = []
        for sp in sorted(per_sp):
            pool.extend(rng.sample(per_sp[sp], min(2, len(per_sp[sp]))))
        rng.shuffle(pool)
        out.extend(pool[:per_part])
    rng.shuffle(out)                      # 부위끼리 뭉치면 순서로 라벨을 추측하게 된다
    return out


def encode(reader, logical: str, px: int) -> str | None:
    img = reader.get(logical)
    if img is None:
        return None
    img.thumbnail((px, px))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html>
<meta charset="utf-8"><title>부위 라벨 감사</title>
<style>
 body{font:15px/1.6 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:24px;
      background:#faf9f7;color:#1a1a1a}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#666;font-size:13px;margin-bottom:20px}
 .card{background:#fff;border:1px solid #e3e0db;border-radius:10px;padding:16px;
       margin-bottom:16px}
 img{width:100%;max-width:360px;border-radius:8px;display:block;margin:0 auto 12px}
 .opts{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
 button.opt{font:14px system-ui;padding:7px 13px;border:1px solid #d0ccc5;background:#fff;
       border-radius:999px;cursor:pointer}
 button.opt:hover{background:#f0eee9} button.opt.sel{background:#1a1a1a;color:#fff;
       border-color:#1a1a1a}
 .bar{position:sticky;top:0;background:#faf9f7;padding:10px 0;border-bottom:1px solid #e3e0db;
      margin-bottom:16px;display:flex;gap:12px;align-items:center}
 .prog{flex:1;height:6px;background:#e3e0db;border-radius:3px;overflow:hidden}
 .prog>i{display:block;height:100%;background:#1a1a1a;width:0}
 #result{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;background:#fff;
      border:1px solid #e3e0db;border-radius:10px;padding:16px}
 .big{font-size:15px;padding:9px 16px;border-radius:8px;border:1px solid #1a1a1a;
      background:#1a1a1a;color:#fff;cursor:pointer}
 @media(prefers-color-scheme:dark){body{background:#171614;color:#eee}
  .card,#result{background:#201f1c;border-color:#35332e} .bar{background:#171614;
  border-color:#35332e} button.opt{background:#201f1c;color:#eee;border-color:#3d3a34}
  button.opt.sel{background:#eee;color:#111} .prog{background:#35332e} .prog>i{background:#eee}}
</style>
<h1>부위 라벨 감사</h1>
<div class="sub">사진의 <b>주피사체</b>가 무엇인지 고르세요. 라벨은 보이지 않습니다 —
 전부 끝내면 대조 결과가 나옵니다. 진행 상황은 브라우저에 자동 저장됩니다.</div>
<div class="bar"><span id="cnt">0/0</span><span class="prog"><i id="pbar"></i></span>
 <button class="big" id="done">결과 보기</button></div>
<div id="list"></div>
<div id="result" hidden></div>
<script>
const ITEMS = __DATA__;
const CHOICES = __CHOICES__;
const KEY = 'part_audit_v1';
let ans = JSON.parse(localStorage.getItem(KEY) || '{}');

const list = document.getElementById('list');
ITEMS.forEach((it, i) => {
  const c = document.createElement('div'); c.className = 'card';
  const im = document.createElement('img'); im.src = 'data:image/jpeg;base64,' + it.b64;
  im.loading = 'lazy'; im.alt = '감사 대상 ' + (i + 1); c.appendChild(im);
  const o = document.createElement('div'); o.className = 'opts';
  CHOICES.forEach(([v, label]) => {
    const b = document.createElement('button'); b.className = 'opt'; b.textContent = label;
    if (ans[it.id] === v) b.classList.add('sel');
    b.onclick = () => {
      ans[it.id] = v; localStorage.setItem(KEY, JSON.stringify(ans));
      [...o.children].forEach(x => x.classList.remove('sel')); b.classList.add('sel');
      draw();
    };
    o.appendChild(b);
  });
  c.appendChild(o); list.appendChild(c);
});

function draw() {
  const n = ITEMS.filter(i => ans[i.id]).length;
  document.getElementById('cnt').textContent = n + '/' + ITEMS.length;
  document.getElementById('pbar').style.width = (n / ITEMS.length * 100) + '%';
}
draw();

document.getElementById('done').onclick = () => {
  const per = {}, conf = {};
  let exact = 0, multi = 0, unclear = 0, wrong = 0, done = 0;
  ITEMS.forEach(it => {
    const a = ans[it.id]; if (!a) return;
    done++;
    per[it.part] = per[it.part] || {n: 0, ok: 0, multi: 0, wrong: 0};
    per[it.part].n++;
    if (a === it.part) { exact++; per[it.part].ok++; }
    else if (a === 'multi') { multi++; per[it.part].multi++; }
    else if (a === 'unclear') { unclear++; }
    else { wrong++; per[it.part].wrong++;
           const k = it.part + ' → ' + a; conf[k] = (conf[k] || 0) + 1; }
  });
  const pct = x => done ? (x / done * 100).toFixed(1) + '%' : '-';
  let s = `판정 ${done}/${ITEMS.length}\n\n`;
  s += `라벨과 일치        ${exact}  (${pct(exact)})\n`;
  s += `여러 기관 동등     ${multi}  (${pct(multi)})   ← 라벨 오류는 아니나 배타적이지 않음\n`;
  s += `다른 기관이 주피사체 ${wrong}  (${pct(wrong)})   ← 실질 라벨 오류\n`;
  s += `판단 불가          ${unclear}  (${pct(unclear)})\n\n부위별\n`;
  Object.keys(per).sort().forEach(p => {
    const d = per[p];
    s += `  ${p.padEnd(7)} n=${String(d.n).padStart(3)}  일치 ${String(d.ok).padStart(3)}` +
         `  동등 ${String(d.multi).padStart(3)}  오류 ${String(d.wrong).padStart(3)}\n`;
  });
  const ck = Object.keys(conf).sort((a, b) => conf[b] - conf[a]);
  if (ck.length) { s += '\n혼동\n'; ck.forEach(k => s += `  ${k}: ${conf[k]}\n`); }
  s += '\n--- 아래를 복사해 전달 ---\n' + JSON.stringify(
      {done, exact, multi, wrong, unclear, per, conf, answers: ans});
  const r = document.getElementById('result'); r.hidden = false; r.textContent = s;
  r.scrollIntoView({behavior: 'smooth'});
};
</script>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="부위 라벨 블라인드 감사 시트 생성.")
    p.add_argument("--rows", default="data/sft/mm_train_resolved.jsonl")
    p.add_argument("--index", default="data/shards/herb_shard_index.json")
    p.add_argument("--shard_dir", default="data/shards")
    p.add_argument("--per_part", type=int, default=20)
    p.add_argument("--px", type=int, default=360)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/part_audit/part_audit.html")
    a = p.parse_args()

    from hanmed_mm.data.shard_image_reader import ShardImageReader
    reader = ShardImageReader(a.index, a.shard_dir)

    items, skipped = [], 0
    for r in sample(a.rows, a.per_part, a.seed):
        b64 = encode(reader, r["image"], a.px)
        if b64 is None:
            skipped += 1
            continue
        items.append({"id": r["image"], "part": r["part"],
                      "species": r["species_ko"], "b64": b64})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    html = (PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False))
                .replace("__CHOICES__", json.dumps(CHOICES, ensure_ascii=False)))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    dist = collections.Counter(i["part"] for i in items)
    print(f"✔ {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print(f"  표본 {len(items)}장 / 종 {len({i['species'] for i in items})}종"
          f"{f' (이미지 없어 제외 {skipped})' if skipped else ''}")
    print(f"  부위별 {dict(sorted(dist.items()))}")
    print("  라벨은 HTML 안에 있지만 판정 전에는 화면에 나오지 않는다(블라인드).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
