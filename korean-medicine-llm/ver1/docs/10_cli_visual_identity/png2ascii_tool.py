"""Quick PNG → ASCII renderer for HanMed_2.png.

Samples the image on a grid matching terminal cell aspect (~2:1 tall:wide),
maps to density characters. Transparent/white pixels become space.
"""
import sys
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "hammed_icon/HanMed_2.png"
target_cols = int(sys.argv[2]) if len(sys.argv) > 2 else 50

# density ramp (dark → light) — using ASCII only
RAMP = "@#8%*+o=-:. "  # darker = more dense on left

img = Image.open(path).convert("RGBA")
W, H = img.size
# terminal cell aspect: char is ~2x taller than wide, so target_rows ~ cols*H/W/2
target_rows = int(target_cols * H / W / 2.1)

# Compose over white bg so alpha doesn't break luminance
bg = Image.new("RGB", img.size, (255, 255, 255))
bg.paste(img, mask=img.split()[-1])
img = bg.resize((target_cols, target_rows), Image.LANCZOS).convert("L")

lines = []
for y in range(target_rows):
    row = []
    for x in range(target_cols):
        lum = img.getpixel((x, y)) / 255  # 0=dark, 1=light
        # Invert: darker pixel → higher density char
        idx = int(lum * (len(RAMP) - 1))
        row.append(RAMP[idx])
    lines.append("".join(row).rstrip())
print("\n".join(lines))
