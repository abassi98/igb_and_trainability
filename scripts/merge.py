from pathlib import Path
import sys

# Allow direct execution with `python scripts/<name>.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plot_style import configure_plot_style

configure_plot_style()

import matplotlib.pyplot as plt
from PIL import Image

# Open both PNG figures
img1 = Image.open("figures/train/mlp_train_main.png")
img2 = Image.open("figures/train/largevit_train_main.png")

# Match heights (resize if needed, optional)
if img1.height != img2.height:
    ratio = img1.height / img2.height
    img2 = img2.resize((int(img2.width * ratio), img1.height))

# Create new canvas for merged image
merged_width = max(img1.width,img2.width)
merged_height = img1.height + img2.height

merged = Image.new("RGB", (merged_width, merged_height), (255, 255, 255))

# Paste images side by side
merged.paste(img1, (0, 0))
merged.paste(img2, (0,img1.height))

# Embed the merged raster image in a tightly cropped PDF.
fig, ax = plt.subplots(figsize=(merged_width / 300, merged_height / 300))
ax.imshow(merged, rasterized=True)
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig("figures/train/main_train.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
