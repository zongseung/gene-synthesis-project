# §10 CLI Visual Identity — Claude Code-style Compact Layout v3

**상태**: draft · 2026-04-20 (01·02 업데이트본 · 사용자 이미지 레퍼런스 반영)
**결정 변경**:
- Mascot 50 col/28 line 버전은 **과도하게 큼** → Claude Code 스타일의 **5~7 line × 10~14 col 소형** 사용
- splash 을 "hero" 아닌 "header + 구분선 + 프롬프트 + 구분선 + statusline" 4 zone 으로 재배치

레퍼런스: 사용자 공유한 Claude Code 스크린샷 (top: orange pixel-art + 4-line text block, mid: divider/prompt/divider, bot: `┌─ cyberdeck ──` statusline with `[NOVA]`, `CTX {·······}` 등 pill).

---

## 1. Mascot 소형 변형 (ANSI truecolor, **alpha-aware · auto-crop**)

**업데이트**: 이전 버전은 투명 PNG 를 흰색 배경 위에 합성해 사각 박스로 렌더됐음. v3.1 부터 원본 `HanMed_2.png` 의 alpha 채널을 그대로 존중 — 투명 픽셀은 space 로 건너뛰어 배경이 터미널 기본색으로 비침. 또한 alpha bbox 로 **자동 크롭** (상하좌우 투명 여백 제거) 해 최소 면적 보장.

| 파일 | 크기 (char 셀) | 용도 |
|---|---|---|
| `turtle_10col.ansi` | 10 × 5 | 정말 작게 (Claude Code 와 동일 면적) |
| **`turtle_12col.ansi`** | **12 × 6** | **권장** — 균형 |
| `turtle_14col.ansi` | 14 × 7 | 살짝 크게 |
| `turtle_16col.ansi` | 16 × 8 | max (header text 4 line 과 정렬 위해 상·하 padding) |

각 cell 렌더 규칙:
- 상·하 픽셀 **모두 투명** → ` ` (공백, 터미널 배경색 노출)
- **둘 다 불투명** → `▀` + FG(상) + BG(하)
- 상만 불투명 → `▀` + FG(상) + BG reset
- 하만 불투명 → `▄` + FG(하) + BG reset

모든 변형은 `png2block_tool.py` 로 재생성:
```bash
/usr/bin/python3 docs/10_cli_visual_identity/png2block_tool.py \
    hammed_icon/HanMed_2.png 12 \
    > docs/10_cli_visual_identity/turtle_12col.ansi
```

ASCII shape preview (색·반투명 제거, 14col 기준):

```
▀▀       ▀ ▀▀
▀▀▀     ▀▀▀▀▀
▀▀ ▄▀▀▀▀▀▀▀▀▀▄
▄▀▀▀▀▀▀▀▀▄▀▀▀▀
▀▀▀▀▀▀▀▀▀▀▀▀
▀▀▀▀▀▀▀▀▀▀▀▀▀▀
▀▀▀▀▀▀▀▀▀▀▀▀▀▀
```

- 상단 3 row = 매달린 약초 3묶음 + 거북 머리 라운드
- 중·하단 4 row = 몸통 + 약연 + 꿀단지 (색으로 분리됨)

## 2. 최종 레이아웃 (Claude Code 스타일, 80 col 타깃)

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   [▀▀▀▀▀▀▀▀▀▀▀▀]   HanMed CLI v0.1                                    │
│   [▀▀▀▀ 거북 ▀▀▀]   DONGUI · Bllossom-8B + r=32 LoRA · P-A+ CPT        │
│   [▀▀▀▀▀▀▀▀▀▀▀▀]   ~/korean-medicine-llm                              │
│   [▀▀▀▀▀▀▀▀▀▀▀▀]   Welcome to DONGUI · /help, /save, /reset, /exit    │
│   [▀▀▀▀▀▀▀▀▀▀▀▀]                                                      │
│   [▀▀▀▀▀▀▀▀▀▀▀▀]                                                      │
│   ─────────────────────────────────────────────────────────────       │
│                                                                       │
│   >                                                                   │
│                                                                       │
│   ─────────────────────────────────────────────────────────────       │
│     ┌─ hanmed ─────                                                   │
│     [DONGUI]  P-A+ CPT adapter   uptime 0s   branch main              │
│     CTX {·················} 0%       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.1 Header (mascot + metadata)

좌측: **12 col × 6 line** turtle (ANSI truecolor). 우측: 4 line text block.

| 라인 | 내용 | 스타일 |
|---|---|---|
| 1 | `HanMed CLI v0.1` | bold white / 기본 |
| 2 | `DONGUI · Bllossom-8B + r=32 LoRA · P-A+ CPT` | dim (grey70) |
| 3 | `~/korean-medicine-llm` | dim cyan |
| 4 | `Welcome to DONGUI · /help, /save, /reset, /exit` | soft amber (dark_orange3) |

- Mascot 6 line > text 4 line 이므로 mascot 아래 2 line 은 공백으로 padding.
- Mascot 과 text 사이 공백: 3 column.

### 2.2 Divider + Prompt

```
  ─────────────────────────────────────────────────────────────

  >

  ─────────────────────────────────────────────────────────────
```

- Divider: `─` × 63 column (유니코드 `U+2500`). ASCII fallback = `-` × 63.
- Prompt: `> ` + 커서 (`▌` or terminal default block).
- 구분선 색: `grey58` (더 은은).

### 2.3 Statusline (맨 아래, 3 line)

```
  ┌─ hanmed ─────
  [DONGUI]  P-A+ CPT adapter   uptime 0s   branch main
  CTX {·················} 0%
```

| 요소 | 의미 | 색 |
|---|---|---|
| `┌─ hanmed ─────` | 모듈 라벨 (모서리 장식) | `grey58` |
| `[DONGUI]` | 현 세션/모델 태그 | magenta bold |
| `P-A+ CPT adapter` | 어댑터 경로 요약 | bold white |
| `uptime 0s` | 세션 경과 | grey70 |
| `branch main` | git branch | cyan |
| `CTX {···} 0%` | 컨텍스트 사용률 (tokens 기준) | cyan |

- 도트 `···`는 채움률에 따라 `▓` 로 바뀜: `CTX {▓▓▓▓▓▓··········} 32%`.
- v3 에서 `VRAM`·`MIX` pill 은 **생략** (정보 과잉). 필요해지면 `/status` 명령어로 별도 호출.

## 3. 사용자 입력 후 스크롤 동작

대화가 진행되면 header + 상단 divider 는 **한 번만 출력하고 스크롤 아웃**. statusline 은 **고정** 되는 게 이상적이지만, rich/alacritty 에서 실제로 splitscreen 고정은 curses 레벨 작업 필요 → v3 에서는:
- **Option X (간단)**: statusline 을 첫 출력 후 유지 안 함. 매 답변 완료 후 `/status` 명령으로 재출력.
- **Option Y (고도화)**: `rich.live.Live` 로 하단 footer 갱신 (추가 공수).

권장: **Option X** (단순성). `rich.live` 도입은 별도 round.

## 4. 실제 스크립트 동작 (mock sequence)

```
┌─ (초기 splash) ──────────────────────────────────────────────
│   (mascot 12col 6line)   HanMed CLI v0.1
│                          DONGUI · Bllossom-8B + r=32 LoRA · P-A+ CPT
│                          ~/korean-medicine-llm
│                          Welcome to DONGUI · /help, /save, /reset, /exit
│
│   ──────────────────────────────────────────────────────────
│   >  동의보감 저자는?
│   ──────────────────────────────────────────────────────────
│
│   [DONGUI] 『東醫寶鑑』은 조선 선조의 명을 받아 어의 허준(許浚)이 ...
│
│   ──────────────────────────────────────────────────────────
│   >  ▍
│   ──────────────────────────────────────────────────────────
│     ┌─ hanmed ─────
│     [DONGUI]  P-A+ CPT adapter   uptime 47s   branch main
│     CTX {▓▓·················} 8%
└─────────────────────────────────────────────────────────────
```

- `>` 위·아래 divider 는 **현재 턴 전용**. 답변이 출력되면 다음 turn 의 divider 가 갱신.
- 마스코트는 1 회만 출력, 이후 스크롤 아웃.

## 5. branding.py 최종 수정안 (v1/v2 교체)

```python
# ── Mascot variants ────────────────────────────────────────────
# 24-bit truecolor, 12 col × 6 rows. 파일 로드 (ANSI 바이너리).
_PROMPTS = Path(__file__).parent.parent.parent.parent / "docs/10_cli_visual_identity"
try:
    MASCOT_TURTLE_12 = (_PROMPTS / "turtle_block_12col.ansi").read_text()
except FileNotFoundError:
    MASCOT_TURTLE_12 = ""   # fallback

# ── Header block ──────────────────────────────────────────────
CLI_TITLE = f"{DISPLAY_NAME} CLI v0.1"
CLI_SUBTITLE = "DONGUI · Bllossom-8B + r=32 LoRA · P-A+ CPT"
CLI_WELCOME = f"Welcome to {DISPLAY_NAME} · /help, /save, /reset, /exit"

# ── Dividers ──────────────────────────────────────────────────
DIVIDER_UNICODE = "─" * 63
DIVIDER_ASCII = "-" * 63

# ── Statusline ────────────────────────────────────────────────
STATUSLINE_TAG = "┌─ hanmed ─────"
```

render.py 에 `render_compact_splash(cwd, adapter, uptime, vram, mix)` 함수 신설해 위 4 zone 을 구성.

## 6. 폐기 (v1·v2 내용 중)

- ❌ **Hero splash** (§3.1 v2 38-line) — 너무 큼
- ❌ MASCOT T-A/B/C 수기 버전 — 사용 안 함 (png2block 렌더만 씀)
- ❌ Mascot_CABINET (약장 v1) — deprecated, 코드에서 제거 가능 (하위호환 alias 도 불필요)
- ✅ 유지: DONGUI banner figlet — 단, **default splash 에는 미표시**. `/banner` 명령어 또는 `--banner` 플래그로만.

## 7. 미해결

- Statusline 값 소스:
  - uptime: session 시작 시점 `time.time()` 저장, 출력 시 diff
  - branch: `git rev-parse --abbrev-ref HEAD` · 캐싱
  - CTX %: tokenizer 로 대화 누적 token count / 모델 max_len
- 색 지원 감지: `rich.console.Console().color_system` 으로 `truecolor` / `256` / `standard` 자동 분기.
- 비 TTY fallback: header 1-line 요약 (`HanMed CLI · DONGUI · adapter:cpt_bllossom`) 만 출력, mascot·divider·statusline 생략.

## 8. 다음 단계

1. ✅ `turtle_block_{10,12,14}col.ansi` 생성 완료
2. [ ] `branding.py` 교체 (§5 스니펫)
3. [ ] `render.py` 에 `render_compact_splash()` + statusline 함수 구현
4. [ ] smoke: `.venv/bin/python -m hanmed_cli` → 3 모드 (TTY truecolor / TTY 256 / non-TTY) 각각 검증
5. [ ] statusline 고정 여부 (rich.live vs 1회 출력) 결정
