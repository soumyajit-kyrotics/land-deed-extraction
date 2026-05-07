from ultralytics import YOLO
import os
import cv2
import numpy as np

# Load your trained model
model = YOLO("runs/detect/train-main/weights/best.pt")   # <-- your trained weights

# Input folder
INPUT_DIR = "unseen_images"
OUTPUT_DIR = "runs/detect/train-main/inference"

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = model.predict(
    source=INPUT_DIR,
    conf=0.25,
    save=False,   # we don't need images saved
    verbose=False
)

def parse_yolo_results(results, model):
    all_outputs = []

    for r in results:
        image_path = r.path
        boxes = r.boxes

        detections = []

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        all_outputs.append({
            "image_path": image_path,
            "detections": detections
        })

    return all_outputs


def crop_and_save(image_path, detections, output_root):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    doc_id = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_root, doc_id)
    os.makedirs(save_dir, exist_ok=True)

    # Keep best detection per class
    best = {}

    for det in detections:
        label = det["label"]
        if label not in best or det["confidence"] > best[label]["confidence"]:
            best[label] = det

    for label, det in best.items():
        x1, y1, x2, y2 = det["bbox"]

        # Expand bbox (IMPORTANT for OCR safety)
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(save_dir, f"{label}.png"), crop)

    return save_dir

# Process the results
if __name__ == "__main__":
    parsed_data = parse_yolo_results(results, model)
    
    if not parsed_data:
        print("No results to process.")
        
    for data in parsed_data:
        image_path = data["image_path"]
        detections = data["detections"]
        if detections:
            print(f"Cropping and saving {len(detections)} detections for: {os.path.basename(image_path)}")
            crop_and_save(image_path, detections, OUTPUT_DIR)
        else:
            print(f"No detections found for: {os.path.basename(image_path)}")