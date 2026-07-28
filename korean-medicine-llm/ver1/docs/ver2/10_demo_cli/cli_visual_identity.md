# 10.11 CLI Visual Identity · ASCII Design

> 목적: `hanmed chat` 가 단순한 연구용 콘솔이 아니라, **한의학 문헌 보조 AI** 라는 인상을 첫 화면에서 주도록 한다. v0 는 웹 UI 가 없으므로, 배너·캐릭터·프롬프트 톤이 곧 제품 인상이다.

## 10.11.1 디자인 방향

지향점은 다음 4개:

- **scholarly**: 해커툴보다 문헌실 / 서고 / 약장 분위기
- **warm**: 병원 장비 느낌보다 한약방 / 고서 느낌
- **compact**: 80~100 column 터미널에서 깨지지 않음
- **portable**: macOS Terminal, iTerm2, Ubuntu GNOME, tmux 에서 동일 렌더

R3.5 결론:
- 배너와 캐릭터는 **ASCII only** 로 간다
- 박스 드로잉, Hangul/Hanja 아트, full-width 문자는 v0 에서 쓰지 않는다
- 한글/한문은 배너 아래의 일반 텍스트 줄에서만 쓴다

이유:
- CJK 문자는 터미널마다 폭 계산이 다르다
- 박스 드로잉은 폰트 fallback 에 따라 미세하게 어긋난다
- splash screen 은 예쁘더라도, 한 번 틀어지면 싸구려처럼 보인다

## 10.11.2 Naming 전략

실행 command 와 화면에 보이는 브랜드 이름은 분리:

- **command**: `hanmed`
- **display name**: splash / prompt / help 에만 노출

이렇게 하면:
- 패키지명/스크립트명은 기능적으로 유지
- 화면 표현은 더 창의적으로 가져갈 수 있음
- 나중에 `serve` 나 웹 UI 가 생겨도 display name 만 확장 가능

## 10.11.3 이름 후보

| 이름 | 어감 | 장점 | 단점 | 판정 |
|---|---|---|---|---|
| **HYANG** | 따뜻하고 간결 | `향약` 연상, 5글자, ASCII 배너 예쁨 | 초견에 한의학 즉시 연상은 약함 | 차선 |
| **DONGUI** | 정통·고전 | `동의보감` 계열 연상, 권위감, 첫인상 즉시 명확 | 다소 무겁고 길다 | **추천** |
| **BONCHO** | 본초학 중심 | 약재/materia medica 인상 강함 | QA assistant 전체를 대표하기엔 다소 좁다 | 보류 |
| **MAEK** | 짧고 날카로움 | 기억하기 쉬움 | 맥진/진단 뉘앙스가 강해 safety 상 애매 | 비추천 |

## 10.11.4 추천안

### 기본 추천: `DONGUI`

표기:

- **DONGUI**
- **DONGUI // Korean Medicine AI**
- **DONGUI // HanMed Archive Assistant**

의미:

- `동의(東醫)` 계열의 정통성과 고전 문헌 이미지를 직접적으로 준다
- 처음 보는 사용자도 한의학/문헌 보조 도구라는 방향을 바로 이해하기 쉽다
- 논문 데모, 학회 발표, 연구실 내부 도구에 모두 잘 맞는다

사용 규칙:

- 패키지/커맨드명은 계속 `hanmed`
- 첫 화면, 배너, prompt label 은 `DONGUI` 사용
- 문서 첫 줄에서는 `hanmed` 와 병기: `hanmed chat (display: DONGUI)`

### 차선안: `HYANG`

더 부드럽고 herbal 한 인상을 원하면 `HYANG` 도 여전히 좋은 대안이다. 다만 이번 방향에서는 사용자가 원하는 정통·고전 무드가 더 중요하므로 `DONGUI` 를 우선 채택한다.

## 10.11.5 ASCII 배너 원칙

- 폭: **최대 72 columns**
- 높이: **7~14 lines**
- 문자 집합: `A-Z`, `a-z`, `0-9`, 공백, `. , : ; ' - _ / \ | ( ) [ ]`
- 금지: box drawing, Unicode block, full-width punctuation
- splash 상단에 1회만 출력. 매 turn 반복 금지

## 10.11.6 추천 배너

### Banner A — 추천 (`DONGUI` classic mark)

사용자가 선택한 방향. 이름을 직접 쓰는 wordmark 대신, 더 절제된 상징형 배너로 간다.

```text
  _   _ __   __    _    _   _  ____
 | | | |\ \ / /   / \  | \ | |/ ___|
 | |_| | \ V /   / _ \ |  \| | |  _
 |  _  |  | |   / ___ \| |\  | |_| |
 |_| |_|  |_|  /_/   \_\_| \_|\____|

           KOREAN MEDICINE AI
```

장점:

- pure ASCII
- 어떤 폰트에서도 거의 안 깨짐
- 과한 figlet 느낌이 없고 더 정제돼 보임
- 논문 데모 화면 캡처에 무난함

단점:

- `DONGUI` 라는 이름 자체는 배너에 직접 쓰이지 않음
- 따라서 subtitle / prompt label 에서 브랜드를 보강해야 함

### Banner B — `DONGUI` wordmark 대안

```text
DDDD   OOO   N   N   GGG   U   U  III
D   D O   O  NN  N  G      U   U   I
D   D O   O  N N N  G GG   U   U   I
D   D O   O  N  NN  G  G   U   U   I
DDDD   OOO   N   N   GGG    UUU   III

        KOREAN MEDICINE AI
```

장점:

- 이름이 바로 읽혀서 브랜딩 전달력이 높음

단점:

- 존재감이 강해서 차분한 문헌 툴 느낌은 조금 약해짐

R3.5 추천: **Banner A 채택**, Banner B 는 fallback.

### R3.6 BANNER — `DONGUI` wordmark refined (epic serif-cap)

**디자인 의도**: Banner B 의 block-letter 5-line figlet 이 "더 예쁘게 / 품위있게" 라는 사용자 피드백을 반영해 figlet `epic` font 로 재디자인. parenthesis cap 과 slab-serif 골격이 **고서 활자체** 느낌을 주어 서고·약장 톤에 부합한다. 이름(DONGUI)은 여전히 정면에서 읽히며 높이가 5→8 lines 로 늘어나 §10.11.5 의 최소 7 lines 요구를 충족한다.

```text
 ______   _______  _        _______          _________
(  __  \ (  ___  )( (    /|(  ____ \|\     /|\__   __/
| (  \  )| (   ) ||  \  ( || (    \/| )   ( |   ) (
| |   ) || |   | ||   \ | || |      | |   | |   | |
| |   | || |   | || (\ \) || | ____ | |   | |   | |
| |   ) || |   | || | \   || | \_  )| |   | |   | |
| (__/  )| (___) || )  \  || (___) || (___) |___) (___
(______/ (_______)|/    )_)(_______)(_______)\_______/

                  KOREAN MEDICINE AI
```

측정:

- **width**: 54 columns (≤ 72 ✓)
- **height**: 8 lines (7 ~ 14 ✓)
- **문자 집합**: `' '`, `'('`, `')'`, `'/'`, `'\'`, `'_'`, `'|'` — §10.11.5 허용 집합 부분집합
- **금지 문자**: 없음 (box drawing / Unicode block / full-width punctuation 모두 미사용)
- **subtitle 정렬**: `(54 − 18) / 2 = 18` 공백 padding 으로 `KOREAN MEDICINE AI` 를 wordmark 하단 중앙에 배치

장점:

- parenthesis cap 으로 slab-serif/고서 느낌 부여 — `DONGUI` 존재감을 유지하면서도 block-letter 보다 정제됨
- 5 → 8 lines 로 splash 비중이 자연스러워져 적정 proportion
- 오직 7 종 ASCII 문자만 사용 → 모든 터미널 (macOS/iTerm2, Ubuntu, tmux) 에서 동일 렌더 보장

단점:

- 54 columns 로 폭이 Banner B 대비 넓음 — 좁은 SSH 세션 (< 60 col) 에서는 wrap 가능성

**채택 결론 (R3.6)**: `src/hanmed_cli/prompts/branding.py` 의 `BANNER` 상수를 이 아트로 교체. Banner A (상징형) 는 reserved fallback 으로 유지.

## 10.11.7 ASCII 캐릭터 / 마스코트

배너 아래에 작은 마스코트를 붙이면 CLI 가 덜 딱딱해진다. 다만 귀여움보다 **고전 서고 + 약재 보조자** 톤이 중요하다.

### Mascot A — Scholar Apothecary (추천)

```text
            .-"""-.
           /  .-.  \
          |  (o o)  |
          |   \_/   |
          /'-.___.-'\
         /  /| | |\  \
        /__/ | | | \__\
            /_/ \_\
             /| |\
            /_| |_\
```

의도:

- 사람처럼 보이되 너무 만화같지 않음
- scholar + apothecary 사이의 중간 톤
- 약간의 의인화는 있지만 임상 캐릭터처럼 보이지 않음

### Mascot B — Herbal Cabinet

```text
      .--------------------.
     / .------------------. \
    / /  []  []  []  []   \ \
   | |   []  []  []  []    | |
   | |        ____         | |
   | |       / __ \        | |
   | |       \____/        | |
    \ \                    / /
     '--------------------'
```

의도:

- 사람 캐릭터 대신 약장 이미지
- 더 정적이고 문헌실 분위기

R3.5 추천: `DONGUI` 조합에서는 **Mascot B** 가 더 잘 맞고, 친근한 REPL 톤을 원하면 **Mascot A** 도 가능.

## 10.11.8 시작 화면 조합안

### 최종 추천 레이아웃

```text
  _   _ __   __    _    _   _  ____
 | | | |\ \ / /   / \  | \ | |/ ___|
 | |_| | \ V /   / _ \ |  \| | |  _
 |  _  |  | |   / ___ \| |\  | |_| |
 |_| |_|  |_|  /_/   \_\_| \_|\____|

           KOREAN MEDICINE AI

      .--------------------.
     / .------------------. \
    / /  []  []  []  []   \ \
   | |   []  []  []  []    | |
   | |        ____         | |
   | |       / __ \        | |
   | |       \____/        | |
    \ \                    / /
     '--------------------'

  HanMed classical text assistant
  KIOM mediclassics.kr based training
  Not for clinical decision-making

  /help  /save  /reset  /exit
```

이 조합의 장점:

- 첫인상이 더 차분하고 세련됨
- 바로 아래 문구에서 academic / safety tone 을 잡아줌
- 과장된 “AI 쇼” 느낌 없이 품위가 있음

## 10.11.9 Prompt label / 상태 문구

배너만 예쁘고 prompt 가 평범하면 인상이 약해진다. 아래처럼 통일:

- user prompt: `[you]`
- assistant prompt: `[dongui]`
- error: `[dongui:error]`
- safety refusal: `[dongui:safe]`

예:

```text
[you] 인삼의 성미와 귀경 알려줘
[dongui] 인삼은 맛이 달고 약간 쓰며...
```

`[hanmed]` 도 나쁘지 않지만, 시각 아이덴티티까지 고려하면 `[dongui]` 가 더 정돈돼 보인다.

## 10.11.10 색감 가이드 (선택)

v0 는 ASCII-only 가 기본이지만, `rich` 를 쓰므로 색은 아주 절제해서 넣을 수 있다.

- title: soft amber
- mascot: muted sage
- metadata / footer: dim parchment
- refusal / warning: red 대신 **burnt orange**

중요:

- 형광색 금지
- 진보라/네온 청록 같은 “generic AI” 색감 금지
- black background / white background 둘 다 읽혀야 함

## 10.11.11 구현 규칙

실제 코드 구현 시:

1. splash art 는 `src/hanmed_cli/render.py` 또는 `prompts/branding.py` 같은 별도 파일로 분리
2. `textwrap.dedent()` 후 그대로 출력
3. trailing whitespace 제거 금지 — ASCII 정렬 깨질 수 있음
4. `--plain` 옵션에서는 배너 생략 가능
5. non-TTY (`stdout` redirect) 에서는 자동으로 splash 생략

## 10.11.12 최종 결정

v0 디자인 결정 (R3.5 사용자 수정):

- **command**: `hanmed`
- **display name**: `DONGUI`
- **banner**: `Banner B` (DONGUI wordmark — 사용자 채택, R3.5)
- **mascot**: `Mascot B`
- **subtitle**: `KOREAN MEDICINE AI`
- **prompt label**: `[dongui]`

한 줄 요약:

> `hanmed` 는 기능 이름으로 남기고, 사용자가 실제로 보는 얼굴은 **DONGUI** 로 설계한다. 해커툴보다 **고전 문헌실의 조용한 보조자** 에 가깝게 보이게 한다.
