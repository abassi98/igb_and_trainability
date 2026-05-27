from PIL import Image

# Open both PNG figures
img1 = Image.open("figures/train/mlp_train_main.png")
img2 = Image.open("figures/train/largevit_train_main.png")

# Ensure same DPI (Pillow stores DPI in info dictionary)
dpi = img1.info.get("dpi", (300, 300))  # default 300 if missing

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

# Save with DPI preserved
merged.save("figures/train/main_train.png", dpi=dpi*2)
