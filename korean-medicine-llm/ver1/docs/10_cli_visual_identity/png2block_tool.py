"""PNG → half-block + 24-bit truecolor · transparent-aware + auto-crop.

- alpha bbox 로 자동 크롭 (투명 여백 제거)
- 각 cell 상·하 픽셀 alpha 별 처리:
    both transparent → " " (색 없음)
    both opaque      → "▀" (FG=top, BG=bottom)
    only top opaque  → "▀" (FG=top, BG reset)
    only bot opaque  → "▄" (FG=bottom, BG reset)
"""
import sys
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "hammed_icon/HanMed_2.png"
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 14
alpha_thresh = int(sys.argv[3]) if len(sys.argv) > 3 else 16  # below = transparent

img = Image.open(path).convert("RGBA")
bbox = img.getchannel("A").getbbox()
if bbox:
    img = img.crop(bbox)

W, H = img.size
rows = int(cols * H / W)
if rows % 2:
    rows += 1
# NEAREST: 본 프로젝트 마스코트는 pixel art (예: 24×24 native 를 20× 업스케일한 480×480).
# LANCZOS 는 픽셀 경계를 blur 시켜 crispness 손실 → nearest 로 원본 그리드 샘플링 유지.
img = img.resize((cols, rows), Image.NEAREST)
px = img.load()

RESET = "\x1b[0m"
lines = []
for y in range(0, rows, 2):
    line = []
    for x in range(cols):
        tr, tg, tb, ta = px[x, y]
        br, bg, bb, ba = px[x, y + 1]
        top_on = ta >= alpha_thresh
        bot_on = ba >= alpha_thresh
        if not top_on and not bot_on:
            line.append(" ")
            continue
        if top_on and bot_on:
            line.append(
                f"\x1b[38;2;{tr};{tg};{tb}m"
                f"\x1b[48;2;{br};{bg};{bb}m▀{RESET}"
            )
        elif top_on:  # only top opaque
            line.append(f"\x1b[38;2;{tr};{tg};{tb}m▀{RESET}")
        else:  # only bottom opaque
            line.append(f"\x1b[38;2;{br};{bg};{bb}m▄{RESET}")
    # trim trailing spaces to keep lines tight
    s = "".join(line).rstrip()
    lines.append(s)
print("\n".join(lines))
