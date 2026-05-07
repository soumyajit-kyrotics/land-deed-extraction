import os
import cv2
import re
import json
import glob
import argparse
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/train-main/weights/best.pt"
VLM_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
INPUT_DIR = "unseen_images"
OUTPUT_DIR = "pipeline_output"
OCR_REGIONS = ["left_margin", "header", "signature", "stamp"]
DOC_TYPE_REGIONS = ["header", "left_margin", "signature", "stamp", "main_body"]

CORE_FIELDS = [
    "district",
    "registration_office",
    "book_type",
    "year",
    "volume_number",
    "deed_number",
    "transaction_type_major",
    "transaction_type_minor",
    "page_from",
    "page_to",
    "additional_pages",
    "document_type",
    "consideration_amount",
]

REGION_WEIGHT = {
    "header": 1.0,
    "left_margin": 0.95,
    "signature": 0.65,
    "stamp": 0.6,
}

BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("Loading YOLO on CPU...")
yolo_model = YOLO(MODEL_PATH).to('cpu')

print("Loading VLM via Transformers (Memory Efficient Mode)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
vlm_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Use the official HF v5 loading path for PaddleOCR-VL-1.5.
processor = AutoProcessor.from_pretrained(VLM_ID)
vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_ID,
    dtype=vlm_dtype,
).to(device).eval()

# =========================
# STEP 1: PARSE YOLO OUTPUT
# =========================
def parse_yolo_results(results, model):
    all_outputs = []
    for r in results:
        image_path = r.path
        boxes = r.boxes
        detections = []
        if boxes is None: continue

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

# =========================
# STEP 2: CROP REGIONS
# =========================
def crop_and_save(image_path, detections, output_root):
    img = cv2.imread(image_path)
    if img is None:
        print(f"   [WARN] Could not read image: {image_path}")
        return None
    h, w = img.shape[:2]

    doc_id = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_root, doc_id)
    os.makedirs(save_dir, exist_ok=True)

    best = {}
    for det in detections:
        label = det["label"]
        if label not in best or det["confidence"] > best[label]["confidence"]:
            best[label] = det

    if not best:
        print("   [WARN] No detections to crop.")
        return save_dir

    print(f"   [INFO] Saving {len(best)} best region crops to: {save_dir}")
    for label, det in best.items():
        x1, y1, x2, y2 = det["bbox"]
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(save_dir, f"{label}.png"), crop)

    return save_dir

# =========================
# STEP 4: OCR (Transformers Logic)
# =========================
def run_ocr_on_regions(doc_dir):
    results = {}
    prompt = "OCR:"
    
    for region in OCR_REGIONS:
        path = os.path.join(doc_dir, f"{region}.png")
        if not os.path.exists(path):
            print(f"   [INFO] Region not found, skipping OCR: {region}")
            continue

        # Convert to PIL for Transformers
        raw_image = Image.open(path).convert("RGB")
        
        # Prepare multimodal chat-template inputs per PaddleOCR-VL HF usage.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # Decode only generated continuation tokens.
        prompt_len = inputs["input_ids"].shape[-1]
        result_text = processor.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
        results[region] = result_text
        print(f"   [INFO] OCR done for {region} (chars={len(result_text)})")
        
    return results


def generate_from_image(raw_image, prompt, max_new_tokens=512):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()


def normalize_doc_type_label(text):
    t = text.strip().lower()
    if "mixed" in t or "both" in t:
        return "mixed"
    if "handwritten" in t or "hand written" in t:
        return "handwritten"
    if "printed" in t or "typed" in t:
        return "printed"
    return None


def classify_document_type(doc_dir):
    prompt = (
        "You are a document-style classifier. Do NOT transcribe text. "
        "Return exactly one lowercase token from this set: printed, handwritten, mixed."
    )
    votes = []
    evidence_bits = []

    for region in DOC_TYPE_REGIONS:
        path = os.path.join(doc_dir, f"{region}.png")
        if not os.path.exists(path):
            continue
        raw_image = Image.open(path).convert("RGB")
        label_raw = generate_from_image(raw_image, prompt, max_new_tokens=24)
        label = normalize_doc_type_label(label_raw)
        if label is None:
            # Fallback when model returns OCR-like text instead of a class label.
            if region in ("header", "left_margin", "stamp"):
                label = "printed"
            elif region == "signature":
                label = "handwritten"
            elif region == "main_body":
                label = "handwritten"
        if label:
            votes.append((region, label, label_raw))
            evidence_bits.append(f"{region}:{label_raw}")

    if not votes:
        return None

    counts = {"printed": 0, "handwritten": 0, "mixed": 0}
    for _, label, _ in votes:
        counts[label] += 1

    chosen = max(counts, key=counts.get)
    total = len(votes)
    confidence = counts[chosen] / total

    # If both printed and handwritten appear in votes, force mixed.
    if counts["printed"] > 0 and counts["handwritten"] > 0:
        chosen = "mixed"
        confidence = max(confidence, (counts["printed"] + counts["handwritten"]) / total)

    confidence = clip_confidence(confidence)
    source_region = votes[0][0] if len(votes) == 1 else "multi_region"
    return {
        "value": chosen,
        "original_value": chosen,
        "confidence": confidence,
        "source_region": source_region,
        "evidence_text": " | ".join(evidence_bits),
    }

# =========================
# STEP 5: FIELD EXTRACTION
# =========================
def normalize_bengali_digits(text):
    if text is None:
        return None
    return text.translate(BENGALI_DIGITS)


def empty_field_payload():
    return {
        "value": None,
        "original_value": None,
        "confidence": 0.0,
        "source_region": None,
        "evidence_text": None,
    }


def build_field_patterns():
    return {
        "district": [
            (r"(?:district|জেলা)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF .\-]+)", False, 0.95),
        ],
        "registration_office": [
            (r"(?:registration\s*office|reg(?:istration)?\.?\s*office|RO|রেজিস্ট্রেশন\s*অফিস)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF0-9 .,\-/]+)", False, 0.95),
            (r"(?:sub[\s\-]*registrar|s\.?\s*r\.?)\s*[,:\-]\s*([A-Za-z\u0980-\u09FF .\-]+)", False, 0.9),
        ],
        "book_type": [
            (r"(?:book(?:\s*no\.?)?\s*type|book\s*no\.?|বই(?:\s*নং)?)\s*[:\-]?\s*(?:roman\s*)?([IVX]{1,4})\b", False, 0.96),
            (r"(?:book(?:\s*no\.?)?\s*type|book\s*no\.?|বই(?:\s*নং)?)\s*[:\-]?\s*([A-Za-z0-9\-]+)", False, 0.82),
        ],
        "year": [
            (r"(?:year|year\s*of\s*registration|সাল|বছর)\s*[:\-]?\s*((?:19|20)[0-9০-৯]{2})", True, 0.96),
            (r"\b((?:19|20)[0-9০-৯]{2})\b", True, 0.6),
        ],
        "volume_number": [
            (r"(?:volume(?:\s*no\.?|number)?|vol(?:\.|ume)?(?:\s*no\.?)?|খণ্ড)\s*[:\-]?\s*([A-Za-z\-]*[0-9০-৯][A-Za-z0-9০-৯\-]*)", True, 0.95),
        ],
        "deed_number": [
            (r"(?:deed\s*(?:no|number)|দলিল\s*(?:নং|নম্বর)?)\s*[:\-]?\s*([0-9০-৯A-Za-z\-\/]+)", True, 0.97),
            (r"(?:being\s*no\.?)\s*[:\-]?\s*([0-9০-৯]{2,})", True, 0.9),
            (r"(?:no|নং)\s*[:\-]?\s*([0-9০-৯]{2,})", True, 0.55),
        ],
        "transaction_type_major": [
            (r"(?:transaction\s*type\s*\(?major\)?|major\s*head)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.88),
        ],
        "transaction_type_minor": [
            (r"(?:transaction\s*type\s*\(?minor\)?|minor\s*head)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.88),
        ],
        "page_from": [
            (r"(?:page\s*from|starting\s*no\.?)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.9),
        ],
        "page_to": [
            (r"(?:page\s*to|end\s*no\.?)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.9),
        ],
        "additional_pages": [
            (r"(?:additional\s*pages?|অতিরিক্ত\s*পৃষ্ঠা)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.87),
        ],
        "document_type": [
            (r"(?:document\s*type)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.85),
            (r"\b(xerox|hand\s*written|handwritten|typed|টাইপ|হাতের\s*লেখা)\b", False, 0.7),
        ],
        "consideration_amount": [
            (r"(?:consideration\s*amount|transacted\s*amount|মূল্য|টাকা|amount)\s*[:\-]?\s*(?:rs\.?|inr|টাকা)?\s*([0-9০-৯,\.]+)", True, 0.9),
        ],
    }


FIELD_PATTERNS = build_field_patterns()

FIELD_REGION_PRIORITY = {
    "district": ["header", "left_margin", "signature", "stamp"],
    "registration_office": ["header", "left_margin", "signature", "stamp"],
    "book_type": ["header", "left_margin", "signature", "stamp"],
    "year": ["header", "left_margin", "signature", "stamp"],
    "volume_number": ["header", "left_margin", "signature", "stamp"],
    "deed_number": ["left_margin", "header", "signature", "stamp"],
    "transaction_type_major": ["header", "left_margin", "signature", "stamp"],
    "transaction_type_minor": ["header", "left_margin", "signature", "stamp"],
    "page_from": ["header", "left_margin", "signature", "stamp"],
    "page_to": ["header", "left_margin", "signature", "stamp"],
    "additional_pages": ["header", "left_margin", "signature", "stamp"],
    "document_type": ["header", "left_margin", "signature", "stamp"],
    "consideration_amount": ["left_margin", "header", "signature", "stamp"],
}


def clip_confidence(value):
    return round(max(0.0, min(1.0, value)), 2)


def infer_confidence(base_score, region, extracted_value):
    region_boost = REGION_WEIGHT.get(region, 0.6)
    quality_boost = 0.0
    if extracted_value:
        l = len(extracted_value.strip())
        if 2 <= l <= 30:
            quality_boost += 0.05
        if re.search(r"[A-Za-z\u0980-\u09FF]", extracted_value):
            quality_boost += 0.02
    return clip_confidence((base_score * region_boost) + quality_boost)


def select_best_match_for_field(field, ocr_texts):
    priorities = FIELD_REGION_PRIORITY[field]
    patterns = FIELD_PATTERNS[field]
    best = None

    for region in priorities:
        text = ocr_texts.get(region)
        if not text:
            continue
        for pattern, is_numeric, base_score in patterns:
            m = re.search(pattern, text, flags=re.I)
            if not m:
                continue
            original_value = m.group(1).strip() if m.lastindex else m.group(0).strip()
            normalized_value = normalize_bengali_digits(original_value) if is_numeric else original_value
            confidence = infer_confidence(base_score, region, normalized_value)
            evidence = m.group(0).strip()
            candidate = {
                "value": normalized_value,
                "original_value": original_value,
                "confidence": confidence,
                "source_region": region,
                "evidence_text": evidence,
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
    return best


def extract_fields(ocr_texts):
    fields = {field: empty_field_payload() for field in CORE_FIELDS}

    # Special-case robust page range extraction from forms like "PAGES 55 TO 57".
    page_range_pattern = re.compile(
        r"(?:pages?|পৃষ্ঠা)\s*([0-9০-৯]+)\s*(?:to|TO|\-)\s*([0-9০-৯]+)",
        flags=re.I,
    )
    for region in FIELD_REGION_PRIORITY["page_from"]:
        text = ocr_texts.get(region)
        if not text:
            continue
        m = page_range_pattern.search(text)
        if not m:
            continue
        p_from_orig = m.group(1).strip()
        p_to_orig = m.group(2).strip()
        fields["page_from"] = {
            "value": normalize_bengali_digits(p_from_orig),
            "original_value": p_from_orig,
            "confidence": infer_confidence(0.96, region, p_from_orig),
            "source_region": region,
            "evidence_text": m.group(0).strip(),
        }
        fields["page_to"] = {
            "value": normalize_bengali_digits(p_to_orig),
            "original_value": p_to_orig,
            "confidence": infer_confidence(0.96, region, p_to_orig),
            "source_region": region,
            "evidence_text": m.group(0).strip(),
        }
        break

    for field in CORE_FIELDS:
        if fields[field]["value"] is not None:
            continue
        match = select_best_match_for_field(field, ocr_texts)
        if match:
            fields[field] = match
    return fields


def to_legacy_flat_fields(structured_fields):
    legacy = {}
    for field in CORE_FIELDS:
        legacy[field] = structured_fields.get(field, {}).get("value")
    return legacy

# =========================
# STEP 6: PROCESS
# =========================
def process_document(image_path, legacy_flat_output=False):
    print(f"\n[DOC] Processing: {os.path.basename(image_path)}")
    # 1. Detection (CPU)
    res = yolo_model(image_path, verbose=False)
    parsed = parse_yolo_results(res, yolo_model)
    if not parsed:
        print("   [WARN] Empty YOLO output.")
        return None
    print(f"   [INFO] YOLO detections: {len(parsed[0]['detections'])}")

    # 2. Cropping
    doc_dir = crop_and_save(parsed[0]["image_path"], parsed[0]["detections"], OUTPUT_DIR)
    if not doc_dir:
        print("   [WARN] Crop step failed.")
        return None

    # 3. OCR (VLM GPU - Half Precision)
    ocr_texts = run_ocr_on_regions(doc_dir)
    found_regions = sorted(ocr_texts.keys())
    missing_regions = sorted([r for r in OCR_REGIONS if r not in ocr_texts])
    print(f"   [INFO] OCR regions found: {found_regions if found_regions else 'none'}")
    print(f"   [INFO] OCR regions missing: {missing_regions if missing_regions else 'none'}")
    if not ocr_texts:
        print("   [WARN] No OCR text extracted from expected non-main_body regions.")

    structured_fields = extract_fields(ocr_texts)
    doc_type_payload = classify_document_type(doc_dir)
    if doc_type_payload is not None:
        structured_fields["document_type"] = doc_type_payload
        print(
            f"   [FIELD] document_type: value={doc_type_payload['value']!r} "
            f"(conf={doc_type_payload['confidence']}, region={doc_type_payload['source_region']})"
        )

    unresolved = [f for f, payload in structured_fields.items() if payload["value"] is None]
    resolved = [f for f in CORE_FIELDS if f not in unresolved]
    print(f"   [INFO] Extracted fields: {resolved if resolved else 'none'}")
    print(f"   [INFO] Unresolved fields: {unresolved if unresolved else 'none'}")

    for field in resolved:
        payload = structured_fields[field]
        print(
            f"   [FIELD] {field}: value={payload['value']!r} "
            f"(orig={payload['original_value']!r}, conf={payload['confidence']}, region={payload['source_region']})"
        )

    fields = to_legacy_flat_fields(structured_fields) if legacy_flat_output else structured_fields
    return {"doc_id": os.path.basename(image_path), "fields": fields}

def main():
    parser = argparse.ArgumentParser(description="YOLO + PaddleOCR-VL pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N discovered images (for quick testing).",
    )
    parser.add_argument(
        "--legacy-flat-output",
        action="store_true",
        help="Emit flat field values instead of structured payload objects.",
    )
    args = parser.parse_args()

    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))
    image_paths = sorted(set(image_paths))
    all_results = []

    if not image_paths:
        print(f"[ERROR] No images found in: {os.path.abspath(INPUT_DIR)}")
        return

    print(f"[INFO] Found {len(image_paths)} images in {INPUT_DIR}")
    if args.limit is not None:
        if args.limit <= 0:
            print(f"[ERROR] --limit must be > 0, got: {args.limit}")
            return
        image_paths = image_paths[:args.limit]
        print(f"[INFO] Applying limit: processing first {len(image_paths)} image(s)")
    
    for img_path in image_paths:
        try:
            result = process_document(img_path, legacy_flat_output=args.legacy_flat_output)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[INFO] Completed. Processed={len(all_results)}/{len(image_paths)}")
    print(f"[INFO] Results saved to: {output_path}")

if __name__ == "__main__":
    main()
