import json
import os
import shutil
from PIL import Image

# -------- PATHS --------
COCO_JSON = "annotations/instances_default.json"
IMAGE_DIR = "images"
OUTPUT_DIR = "yolo_dataset"

IMAGES_OUT = os.path.join(OUTPUT_DIR, "images")
LABELS_OUT = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(LABELS_OUT, exist_ok=True)

# -------- VALID CLASSES (FINAL TRAINING SET) --------
VALID_CLASSES = [
    "main_body",
    "left_margin",
    "header",
    "stamp",
    "signature",
    "fingerprint"
]

cat_name_to_index = {name: i for i, name in enumerate(VALID_CLASSES)}

# -------- LOAD COCO --------
with open(COCO_JSON) as f:
    coco = json.load(f)

# Build fast lookup: category_id -> name
cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

# Map image_id -> image info
images = {img["id"]: img for img in coco["images"]}

# Group annotations per image
ann_per_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    ann_per_image.setdefault(img_id, []).append(ann)

# -------- PROCESS IMAGES --------
for img_id, img_info in images.items():
    filename = img_info["file_name"]
    img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {filename}")
        continue

    # Load image for size
    image = Image.open(img_path)
    width, height = image.size

    # Copy image (preserves original quality)
    shutil.copy(img_path, os.path.join(IMAGES_OUT, filename))

    label_file = os.path.join(
        LABELS_OUT,
        filename.rsplit(".", 1)[0] + ".txt"
    )

    with open(label_file, "w") as f:
        for ann in ann_per_image.get(img_id, []):
            cat_id = ann["category_id"]
            category_name = cat_id_to_name[cat_id]

            # -------- FILTER UNWANTED LABELS --------
            if category_name not in VALID_CLASSES:
                continue

            class_id = cat_name_to_index[category_name]

            x, y, w, h = ann["bbox"]

            # -------- CONVERT TO YOLO FORMAT --------
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Clamp values (safety for bad boxes)
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            w_norm = min(max(w_norm, 0), 1)
            h_norm = min(max(h_norm, 0), 1)

            f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

print("✅ Conversion complete!")
print(f"📁 YOLO dataset created at: {OUTPUT_DIR}")