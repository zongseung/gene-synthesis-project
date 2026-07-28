# §10 CLI Visual Identity — Turtle Apothecary Draft (v2)

**상태**: draft · 2026-04-20
**기존 버전**: `src/hanmed_cli/prompts/branding.py` (v1: DONGUI figlet + Mascot B 약장)
**변경 동기**: 신규 마스코트 PNG 3종 (`/home/user/gene-synthesis-project/hammed_icon/HanMed_{1,2,3}.png`) 도입. 특히 `HanMed_2.png` 의 **거북 약사(龜 藥師)** 캐릭터를 CLI 첫 화면 hero 로 반영.

---

## 1. 레퍼런스 이미지

- **경로**: `hammed_icon/HanMed_2.png` (picturestyle pixel-art, 640×640)
- **구성 요소** (시계 방향):
  1. **거북 약사** — 중앙, 올리브 그린 머리·등껍질, 베이지 약사복, 양손으로 약연(藥碾) 그릇 잡음
  2. **걸린 약초 묶음 3개** — 상단에 매달림 (왼쪽 연갈색, 중앙·우측 다크브라운)
  3. **꿀단지 (약탕기)** — 오른쪽 하단, 연기 모락
  4. **분홍 뺨** — 거북 양볼
- **상징 매핑 (한의학)**:
  - 거북 = 장수·지혜·四神 중 현무(玄武). 한의 분야에서 자라탕(鱉甲)·귀판(龜板) 등 본초로도 등장
  - 약연 = 약재 분쇄 도구, 제약(製藥)의 상징
  - 매달린 약초 = 건조 본초 (乾草), 약재고 이미지
  - 꿀단지 + 연기 = 전탕(煎湯), 약 달이는 과정
- **톤**: 따뜻한 파스텔 + 레트로 픽셀. 진지하되 친근, 임상 조언 도구가 아닌 "고전 의서 해제 도우미" 정체성과 부합.

## 2. 색상 팔레트 (터미널 렌더링)

| 역할 | 설명 | HEX 근사 | 256-color | ANSI FG |
|---|---|---|---|---|
| shell-dark | 등껍질 · 본체 외곽 | `#3f5031` | 58 | `\e[38;5;58m` |
| body-green | 머리·팔·다리 | `#8fbd6e` | 107 | `\e[38;5;107m` |
| robe-cream | 약사복 | `#e8d9b6` | 222 | `\e[38;5;222m` |
| herb-brown | 매달린 약초 · 꿀단지 | `#6b4a32` | 94 | `\e[38;5;94m` |
| cheek-rose | 분홍 뺨 · 악센트 | `#e99b99` | 174 | `\e[38;5;174m` |
| steam-gray | 김 · 연기 | `#cfcfcf` | 252 | `\e[38;5;252m` |
| bg-default | 터미널 기본 배경 | — | default | 터미널 따름 |

**원칙**: 색상은 `--plain` / non-TTY 에서 자동 비활성화 (기존 render.py 의 TTY 감지 재사용). 컬러 없어도 ASCII 레이아웃이 깨지지 않게 설계.

## 3. 레이아웃 제약

- 터미널 **80 col × 24 row** 기준. splash 는 최대 **72 col × 20 line** 으로 잡아 좁은 창에서도 잘림 없이 수용.
- 기존 §10.11.5 규정(banner 7~14 line / ≤72 col) 승계.
- ASCII-only 로 우선 설계. 유니코드 글리프(■ ▲ ◕)는 선택적 fallback — 일부 터미널이 폭 변칙 대응 실패.

## 4. Mascot 3종 드래프트

### 4.1 Mascot T-A — 미니 아이콘 (prompt label·inline 용, 3 line)

```
  ,--.
 (o  o)
 /~##~\
```

- 역할: 기존 `[dongui]` prompt label 옆 선택적 표시. 혹은 `/help` 화면의 section divider 로 사용.

### 4.2 Mascot T-B — 중간 근접 (splash side panel, 12 line × 22 col)

```
        .-""""-.
      ,' . __ . `.
     /  (●)  (●)  \
    |      ▼       |
    |    \___/     |
    |              |
   _|______________|_
  /##################\
 (####  H A N  ####)
  \##################/
   `----------------'
      ||        ||
```

- 비율: 머리 쪽 5 line + 몸통·약연 4 line + 등껍질 커브 3 line = 12 line.
- `H A N` 자리에 실제로 약연 그릇 심볼 `\====/` 를 넣어도 됨.

### 4.3 Mascot T-C — 풀 신(全景), splash hero, 18 line × 58 col

```
     |             |                         |
     #             ║                         #
    ┌─┐           ╔═╗           ~~~         ┌─┐
    │ │           ║ ║          (   )        │ │
    │#│           ╚═╝         ~~~~~~        │#│
    │#│  herbs                 steam          │#│

                      .----""----.
                    ,'  .      .  `.
                   /   (●)    (●)   \
                  |        ▼         |
                  |      \___/       |
             _,--'                    `--._
          ,-'                              `-.
         /##    ┌──────────────┐           ##\
        (###    │  ⊙  藥  研  ⊙  │           ###)       ___
         \##    └──────────────┘           ##/       ,'  |
          `-._                          _,-'        /    |
              `---.__________________.---'         |  꿀  |
                  ||              ||                \ 단지 |
                 /__\            /__\                `----'
```

- **핵심**: 거북 + 약연 + 걸린 약초 3개 + 꿀단지 한 프레임. 60 col × 18 line 내부 배치.
- 비용: 복잡 → 일부 터미널 font width (예: 일본어 IME 붙은 환경) 에서 정렬 어긋날 수 있음. `--plain` 시 자동 skip.

## 5. Banner 호환성

기존 v1 `DONGUI` figlet (54 col × 8 line) 은 유지. 새 mascot 은 **banner 아래에 side-by-side** 로 배치하거나, 첫 실행시만 mascot 단독, 이후는 compact.

### 옵션 A — Banner + Mascot T-B nested (권장)

```
  ______   _______  _        _______          _________
 (  __  \ (  ___  )( (    /|(  ____ \|\     /|\__   __/       .-""""-.
 | (  \  )| (   ) ||  \  ( || (    \/| )   ( |   ) (        ,' .__ . `.
 | |   ) || |   | ||   \ | || |      | |   | |   | |       / (●)  (●) \
 | |   | || |   | || (\ \) || | ____ | |   | |   | |      |     ▼      |
 | |   ) || |   | || | \   || | \_  )| |   | |   | |      |   \___/    |
 | (__/  )| (___) || )  \  || (___) || (___) |___) (___  _|____________|_
 (______/ (_______)|/    )_)(_______)(_______)\_______/ /##################\
                                                       (####  龜  藥 師 ####)
                  KOREAN  MEDICINE  AI                  \##################/
                                                         `----------------'
                                                            ||        ||
```

- 좌측 54 col banner + 우측 ~22 col mascot = 총 ~78 col (80 col 허용치 내).
- 하단 subtitle `KOREAN MEDICINE AI` 유지, 서체 일관성 확보.

### 옵션 B — Mascot 단독 (첫 실행시만, 이후 banner 생략)

`--first-run` 플래그 또는 `~/.hanmed/seen_splash` 부재 시 옵션 B 로 풀 hero (Mascot T-C), 두 번째 이후는 옵션 A compact.

## 6. Splash 전체 구성 (옵션 A 기준)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ______   _______  _        _______          _________                       │
│ (  __  \ (  ___  )( (    /|(  ____ \|\     /|\__   __/       .-""""-.        │
│ | (  \  )| (   ) ||  \  ( || (    \/| )   ( |   ) (        ,' .__ . `.       │
│ | |   ) || |   | ||   \ | || |      | |   | |   | |       / (●)  (●) \      │
│ | |   | || |   | || (\ \) || | ____ | |   | |   | |      |     ▼      |     │
│ | |   ) || |   | || | \   || | \_  )| |   | |   | |      |   \___/    |     │
│ | (__/  )| (___) || )  \  || (___) || (___) |___) (___  _|____________|_    │
│ (______/ (_______)|/    )_)(_______)(_______)\_______/ /##################\  │
│                                                       (####  龜  藥 師 ####) │
│                  KOREAN  MEDICINE  AI                  \##################/  │
│                                                         `----------------'   │
│                                                            ||        ||      │
│                                                                              │
│  HanMed classical text assistant                                             │
│  KIOM mediclassics.kr · 26 books · P-A+ CPT                                  │
│  Not for clinical decision-making                                            │
│                                                                              │
│  /help   /save   /reset   /exit                                              │
│                                                                              │
│  [you] ▍                                                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                                             (80 col × 24 row)
```

- `[you]` prompt 앞 커서(`▍`)는 기존 render.py 인터랙션 포인트.
- 3번째 footer 라인은 `ver2` 문구("KIOM mediclassics.kr based training") 를 ver4 실측(26 books · P-A+ CPT)으로 업데이트.

## 7. Prompt label 변형

| v1 (현행) | v2 제안 |
|---|---|
| `[you]` | `[you]` (유지) |
| `[dongui]` | `[dongui]` (유지) *or* `[龜]` (한자 단자, 폭 2) |
| `[dongui:error]` | `[龜:error]` or `[dongui:error]` |

권장: 기본 `[dongui]` 유지 + `--mascot` 옵션 켜면 `[龜]` 로 전환. 한자 폭 2라 정렬 이슈 피하려면 default ASCII.

## 8. 구현 계획 (branding.py 수정 목록)

최소 변경으로 반영 (CLAUDE.md "Don't add features beyond what the task requires" 준수):

1. **branding.py 상수 추가**:
   - `MASCOT_T_A` (3-line mini)
   - `MASCOT_T_B` (12-line side)
   - `MASCOT_T_C` (18-line full)
   - 기존 `MASCOT` (약장) 은 `MASCOT_CABINET` 으로 개명 (하위호환 alias 유지)
2. **SPLASH_LAYOUT 상수**: 옵션 A(banner + T-B nested) 를 literal 문자열로 저장.
3. **`INTRO_LINES` 업데이트**: ver4 반영 문구로 교체 (안 2번째 라인 `KIOM mediclassics.kr · 26 books · P-A+ CPT`).
4. **render.py 변경 없음 가정**: TTY 감지 / plain fallback 은 기존 로직 재사용.
5. **팔레트 적용**: 별도 `render_color.py` 없이 render.py 안에 `COLOR` dict 만 추가. non-TTY 시 `""` 반환.
6. **CLI flag**:
   - `--plain`: 기존 유지 (모든 mascot skip, banner 만)
   - `--mascot {t-a|t-b|t-c|cabinet|none}`: 신규. default `t-b`.
   - `--no-splash`: splash 전체 건너뛰고 바로 `[you]`.

## 9. 미해결 · 다음 단계

- **픽셀→ASCII 자동 변환 시도**: `jp2a` / `ascii-art-generator` 로 HanMed_2.png 를 72×24 그리드에 한번 뽑아보고, 4.3 (T-C) 수기안과 비교.
  ```bash
  jp2a --width=58 hammed_icon/HanMed_2.png > /tmp/asciiauto.txt
  ```
- **cols 폭 검증**: `awk '{ print length }' | sort -nu | tail -1` 로 각 mascot 의 실제 최대 폭 측정 (유니코드 글리프 폭 반영).
- **HanMed_1.png / HanMed_3.png** 의 구성도 점검해 아이콘 set 통일 여부 결정.
- **Font 테스트**: macOS Terminal / iTerm2 / Windows Terminal / Gnome Terminal 각각에서 정렬 검증. 가장 문제 되는 글리프는 `●` `■` `▼` — ASCII fallback 으로 각각 `(o)` `[#]` `v` 대체 준비.

## 10. 참조

- 기존 v1: `src/hanmed_cli/prompts/branding.py:22-76`
- 렌더러: `src/hanmed_cli/render.py`
- 원본 아이콘: `/home/user/gene-synthesis-project/hammed_icon/HanMed_2.png`
- CLI 엔트리: `src/hanmed_cli/main.py`
