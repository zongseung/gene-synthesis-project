"""§10.11 CLI Visual Identity — splash art + brand strings.

단일 책임: ASCII art · display name · prompt labels 만 정의. 렌더링은 render.py.

§10.11.12 결정:
    - command      = "hanmed"
    - display_name = "DONGUI"
    - banner       = Banner A (상징형, HYANG figlet 형태)
    - mascot       = Mascot B (약장)
    - subtitle     = "KOREAN MEDICINE AI"
    - prompt label = "[dongui]"

구현 규칙 (§10.11.11):
    1. 문자열을 그대로 상수로 보관. textwrap.dedent() 없이 literal.
    2. trailing whitespace 보존 (정렬 깨짐 방지) — \\ 줄바꿈 대신 개행 literal 유지.
    3. non-TTY / --plain 은 render.py 에서 split.
"""

from __future__ import annotations


COMMAND_NAME = "hanmed"
DISPLAY_NAME = "DONGUI"
SUBTITLE = "KOREAN MEDICINE AI"

# Prompt labels (§10.11.9)
LABEL_USER = "[you]"
LABEL_ASSISTANT = f"[{DISPLAY_NAME.lower()}]"
LABEL_ERROR = f"[{DISPLAY_NAME.lower()}:error]"
LABEL_SAFE = f"[{DISPLAY_NAME.lower()}:safe]"


# Banner B / R3.6 — DONGUI wordmark, serif-capped figlet 'epic' style.
# 선택 이유: parenthesis cap 이 고서 활자(서고·약장) 분위기를 전달하면서도
# letter 대비가 높아 72col/8line 내에서 DONGUI 를 가장 우아하게 판독시킨다.
# 측정: height=8 lines, width=54 columns (§10.11.5 7~14/≤72 충족).
BANNER = (
    " ______   _______  _        _______          _________\n"
    "(  __  \\ (  ___  )( (    /|(  ____ \\|\\     /|\\__   __/\n"
    "| (  \\  )| (   ) ||  \\  ( || (    \\/| )   ( |   ) (   \n"
    "| |   ) || |   | ||   \\ | || |      | |   | |   | |   \n"
    "| |   | || |   | || (\\ \\) || | ____ | |   | |   | |   \n"
    "| |   ) || |   | || | \\   || | \\_  )| |   | |   | |   \n"
    "| (__/  )| (___) || )  \\  || (___) || (___) |___) (___\n"
    "(______/ (_______)|/    )_)(_______)(_______)\\_______/\n"
)


# Mascot B — Herbal Cabinet 약장 (§10.11.7)
MASCOT = r"""      .--------------------.
     / .------------------. \
    / /  []  []  []  []   \ \
   | |   []  []  []  []    | |
   | |        ____         | |
   | |       / __ \        | |
   | |       \____/        | |
    \ \                    / /
     '--------------------'
"""


# Subtitle block under banner.
# BANNER width = 54 cols, SUBTITLE = 18 chars → pad = (54-18)//2 = 18 spaces
# to center-align under the DONGUI wordmark.
SUBTITLE_LINE = f"{' ' * 18}{SUBTITLE}"


# Footer lines (§10.11.8 최종 조합안)
INTRO_LINES = (
    "HanMed classical text assistant",
    "KIOM mediclassics.kr based training",
    "Not for clinical decision-making",
)

SLASH_HINT = "/help  /save  /reset  /exit"
