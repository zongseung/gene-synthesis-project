"""161종 자동 분류 + README 삽입용 markdown 생성.

분류:
  A  한국 한의학 핵심 (동의보감·사상·향약·언해)    → 수집 **최우선**
  B  종합의서·경험방·처방합편 (조선/한국계)       → 수집 권장
  C  조선이 수용한 중국·일본 고전 의서            → CBETA 20% 슬롯 대체 가능
  D  본초·약재·식이 전문                     → 용어·어휘 보강
  E  전문 분과 (전염·부인·침구·외과·맥진·소아)  → 분과 커버리지
  F  수의학·법의·의원 행정 (조선 내의원 등)       → 선택적
  G  비의학 조선 문헌 (연행일기·고사신서 등)      → **제외 권장**
"""

import json
from pathlib import Path

CORE14 = {1, 4, 8, 9, 24, 38, 56, 59, 69, 86, 93, 100, 182, 291}
CORE25_EXT = {7, 44, 46, 47, 49, 54, 60, 70, 71, 94, 139, 183}


def categorize(zh: str, ko: str) -> tuple[str, str]:
    """returns (category_code, reason)"""
    s = (zh or "") + " " + (ko or "")

    # A. 한국 한의학 핵심
    if "諺解" in s or "언해" in s:
        return "A", "조선 한글 번역 의서 (국역 coverage 상위)"
    if "東醫寶鑑" in s:
        return "A", "허준, 조선 대표 의학서"
    if "東醫壽世" in s or "四象" in s:
        return "A", "이제마 사상의학 (한국 고유 체계)"
    if "鄕藥" in s:
        return "A", "조선 향약 (토종 약재·식생)"

    # B. 한국 종합의서·경험방
    if any(k in s for k in ["經驗", "新方", "神方", "奇方", "效方", "秘", "笈"]):
        return "B", "조선 의가 경험방"
    if any(k in s for k in ["醫方", "醫宗", "醫學", "醫門", "醫鑑", "醫略", "醫彙", "醫要",
                              "彙", "總論", "大全", "寶鑑"]):
        return "B", "종합 의학서 / 요약집"
    if any(k in s for k in ["方藥", "濟衆", "處方", "劑", "合編", "集成", "要訣", "廣濟"]):
        return "B", "처방 합편·편찬"

    # C. 중국·일본 고전 의서
    if any(k in s for k in ["素問", "黃帝", "內經", "難經", "靈樞"]):
        return "C", "황제내경 계열 고전"
    if any(k in s for k in ["傷寒", "金匱", "雜病"]):
        return "C", "상한론·금궤요략 계열"
    if any(k in s for k in ["太平聖惠", "千金", "本經", "景岳", "臨證指南", "活人書",
                              "小兒藥證", "仁齋", "脾胃", "壽世", "重訂", "新刊"]):
        return "C", "중국 고전·명청 의서"
    if any(k in s for k in ["丹波", "吉益", "大塚", "東洞", "古書醫言"]):
        return "C", "일본 한방 (토·강호 의가)"
    if "王氏脉" in s or "脉經" in s or "瀕湖脈" in s:
        return "C", "중국 맥학 고전"

    # D. 본초·약재·식이
    if any(k in s for k in ["本草", "藥性", "藥物", "단방", "單方", "蓼", "草"]):
        return "D", "본초학·약재 사전"
    if any(k in s for k in ["食療", "酒", "食"]):
        return "D", "식이·식료"

    # E. 전문 분과
    if any(k in s for k in ["痘", "麻疹", "疫", "瘟"]):
        return "E", "전염병·두창 전문"
    if any(k in s for k in ["胎", "産", "婦", "女", "小兒", "幼"]):
        return "E", "부인과·소아"
    if any(k in s for k in ["鍼", "灸", "神應"]):
        return "E", "침구·경혈"
    if any(k in s for k in ["外科", "瘍", "瘡", "癰", "腫", "癍"]):
        return "E", "외과·종창·피부"
    if any(k in s for k in ["眼", "口齒", "齒", "銀海"]):
        return "E", "안이과·구치"
    if any(k in s for k in ["救急", "急救"]):
        return "E", "구급"
    if any(k in s for k in ["脈", "脉", "診"]):
        return "E", "맥진·진단"

    # F. 수의학·법의·의원 행정
    if any(k in s for k in ["馬醫", "獸", "馬", "牛"]):
        return "F", "수의학"
    if any(k in s for k in ["律", "檢屍", "檢驗", "獄"]):
        return "F", "법의학·율령"
    if any(k in s for k in ["內醫院", "議政府", "藥房", "定例", "式例", "典禮", "惠局",
                              "審藥", "先生案", "軍中"]):
        return "F", "의원 행정·제도"

    # G. 비의학 조선 문헌
    if any(k in s for k in ["燕行日記", "攷事新書", "丁若鏞", "遺事", "漫筆", "古書"]):
        return "G", "비의학 조선 문헌"

    return "?", "미분류"


def main():
    data = json.load(open("data/stats/mediclassics_book_list.json"))
    books = data["books"]

    rows = []
    for b in books:
        bid = b["book_id"]
        zh = b.get("title_zh") or "?"
        ko = b.get("title_ko") or "—"
        # "서적통계" 는 국역 필드 비어서 다른 텍스트가 들어간 케이스 — "—" 로 정규화
        if ko == "서적통계":
            ko = "—"
        cat, reason = categorize(zh, ko)
        state = (
            "✅"
            if bid in CORE14
            else ("🔥" if bid in CORE25_EXT else "")
        )
        rows.append((bid, ko, zh, cat, reason, state))

    # 통계
    from collections import Counter

    cat_count = Counter(r[3] for r in rows)
    collected_by_cat = Counter(r[3] for r in rows if r[5])

    print("=== 분류별 현황 ===")
    for c in "ABCDEFG?":
        total = cat_count.get(c, 0)
        got = collected_by_cat.get(c, 0)
        if total:
            print(f"  {c}  {got:3d} / {total:3d} 수집")

    # markdown 파일 저장 — README 삽입용
    out = Path("data/stats/book_list_161.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("| id | 국역 | 한자 | 분류 | 이유 | 수집 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for bid, ko, zh, cat, reason, state in rows:
            f.write(f"| {bid} | {ko} | {zh} | **{cat}** | {reason} | {state} |\n")
    print(f"\n저장: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
