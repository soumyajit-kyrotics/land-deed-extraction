import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/train-main/weights/best.pt"
VLM_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
INPUT_DIR = "unseen_images"
OUTPUT_DIR = "pipeline_output"
OCR_REGIONS = ["left_margin", "header", "signature", "stamp"]
DOC_TYPE_REGIONS = ["header", "left_margin", "signature", "stamp", "main_body"]
IMAGE_EXTENSIONS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
PDF_EXTENSIONS = ["*.pdf", "*.PDF"]

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
    "full_page": 0.7,
}

BOOK_TYPES = {"I", "II", "III", "IV"}
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
REGISTRATION_OFFICE_ALIASES = {
    "mckligan": "Mekhliganj",
    "mekligan": "Mekhliganj",
    "mekligam": "Mekhliganj",
    "mekligang": "Mekhliganj",
    "mekhliganj": "Mekhliganj",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("Loading YOLO on CPU...")
yolo_model = YOLO(MODEL_PATH).to("cpu")

print("Loading VLM via Transformers (Memory Efficient Mode)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
vlm_dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(VLM_ID)
vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_ID,
    dtype=vlm_dtype,
).to(device).eval()


# =========================
# PATHS / CACHE
# =========================
def safe_id(value):
    stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:90]}_{digest}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cache_path(output_root, doc_id, page_number, kind, region):
    filename = f"{doc_id}__p{page_number:04d}__{kind}__{region}.json"
    return os.path.join(output_root, "cache", filename)


# =========================
# INPUT LOADING
# =========================
def discover_documents(input_dir, output_root, pdf_dpi):
    documents = []

    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    for image_path in sorted(set(image_paths)):
        documents.append({
            "doc_id": safe_id(image_path),
            "source_path": image_path,
            "source_type": "image",
            "pages": [{"page_number": 1, "image_path": image_path}],
        })

    pdf_paths = []
    for pattern in PDF_EXTENSIONS:
        pdf_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    for pdf_path in sorted(set(pdf_paths)):
        doc_id = safe_id(pdf_path)
        pages = render_pdf_pages(pdf_path, output_root, doc_id, pdf_dpi)
        documents.append({
            "doc_id": doc_id,
            "source_path": pdf_path,
            "source_type": "pdf",
            "pages": pages,
        })

    return documents


def render_pdf_pages(pdf_path, output_root, doc_id, dpi):
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Install it with: uv pip install pymupdf"
        ) from exc

    rendered_dir = ensure_dir(os.path.join(output_root, "rendered_pages", doc_id))
    pdf = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(pdf)):
        page_number = page_idx + 1
        output_path = os.path.join(rendered_dir, f"page_{page_number:04d}.png")
        if not os.path.exists(output_path):
            pix = pdf[page_idx].get_pixmap(matrix=matrix, alpha=False)
            pix.save(output_path)
        pages.append({"page_number": page_number, "image_path": output_path})

    pdf.close()
    return pages


# =========================
# MODEL HELPERS
# =========================
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
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()


def ocr_image(image_path, output_root, doc_id, page_number, region):
    path = cache_path(output_root, doc_id, page_number, "ocr", region)
    cached = load_json(path)
    if cached is not None:
        return cached["text"]

    raw_image = Image.open(image_path).convert("RGB")
    text = generate_from_image(raw_image, "OCR:", max_new_tokens=512)
    save_json(path, {"image_path": image_path, "region": region, "text": text})
    return text


def classify_image_style(image_path, output_root, doc_id, page_number, region):
    path = cache_path(output_root, doc_id, page_number, "style", region)
    cached = load_json(path)
    if cached is not None:
        return cached

    prompt = (
        "You are a document-style classifier. Do NOT transcribe text. "
        "Return exactly one lowercase token from this set: printed, handwritten, mixed."
    )
    raw_image = Image.open(image_path).convert("RGB")
    raw_label = generate_from_image(raw_image, prompt, max_new_tokens=24)
    label = normalize_doc_type_label(raw_label)

    if label is None:
        if region in ("header", "left_margin", "stamp"):
            label = "printed"
        elif region in ("signature", "main_body"):
            label = "handwritten"
        else:
            label = "mixed"

    payload = {"region": region, "raw_label": raw_label, "label": label}
    save_json(path, payload)
    return payload


# =========================
# YOLO DETECTION / CROPPING
# =========================
def parse_yolo_results(results, model):
    all_outputs = []
    for r in results:
        boxes = r.boxes
        detections = []
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "label": model.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": list(map(int, box.xyxy[0])),
            })

        all_outputs.append({"image_path": r.path, "detections": detections})
    return all_outputs


def detect_regions(image_path):
    results = yolo_model(image_path, verbose=False)
    parsed = parse_yolo_results(results, yolo_model)
    return parsed[0]["detections"] if parsed else []


def crop_and_save(image_path, detections, output_root, doc_id, page_number):
    img = cv2.imread(image_path)
    if img is None:
        print(f"   [WARN] Could not read image: {image_path}")
        return None

    crop_dir = ensure_dir(os.path.join(output_root, "crops", doc_id, f"page_{page_number:04d}"))
    h, w = img.shape[:2]
    best = {}

    for det in detections:
        label = det["label"]
        if label not in best or det["confidence"] > best[label]["confidence"]:
            best[label] = det

    if not best:
        print("   [WARN] No detections to crop.")
        return crop_dir

    print(f"   [INFO] Saving {len(best)} best region crops to: {crop_dir}")
    for label, det in best.items():
        x1, y1, x2, y2 = det["bbox"]
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(crop_dir, f"{label}.png"), crop)

    return crop_dir


# =========================
# FIELD EXTRACTION
# =========================
def normalize_bengali_digits(text):
    if text is None:
        return None
    return text.translate(BENGALI_DIGITS)


def normalize_doc_type_label(text):
    t = text.strip().lower()
    if "mixed" in t or "both" in t:
        return "mixed"
    if "handwritten" in t or "hand written" in t:
        return "handwritten"
    if "printed" in t or "typed" in t:
        return "printed"
    return None


def normalize_registration_office(value):
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z\u0980-\u09FF ]+", "", value).strip()
    key = cleaned.lower().replace(" ", "")
    return REGISTRATION_OFFICE_ALIASES.get(key, cleaned or value.strip())


def empty_field_payload():
    return {
        "value": None,
        "original_value": None,
        "confidence": 0.0,
        "source_page": None,
        "source_region": None,
        "evidence_text": None,
        "method": None,
        "validation_status": "missing",
    }


def build_field_patterns():
    return {
        "district": [
            (r"\b(?:district|জেলা)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF .\-]+)", False, 0.95),
        ],
        "registration_office": [
            (r"\b(?:registration\s*office|reg(?:istration)?\.?\s*office|রেজিস্ট্রেশন\s*অফিস)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF0-9 .,\-/]+)", False, 0.95),
            (r"\bRO\b\s*[:\-]?\s*([A-Za-z\u0980-\u09FF0-9 .,\-/]+)", False, 0.9),
            (r"\b(?:sub[\s\-]*registrar|s\.?\s*r\.?)\s*[,:\-]\s*([A-Za-z\u0980-\u09FF .\-]+)", False, 0.9),
        ],
        "book_type": [
            (r"\b(?:book(?:\s*no\.?)?\s*type|book\s*no\.?|বই(?:\s*নং)?)\s*[:\-]?\s*(?:roman\s*)?([IVX]{1,4})\b", False, 0.96),
        ],
        "year": [
            (r"\b(?:year|year\s*of\s*registration|সাল|বছর)\s*[:\-]?\s*((?:18|19|20)[0-9০-৯]{2})", True, 0.96),
            (r"\b((?:18|19|20)[0-9০-৯]{2})\b", True, 0.55),
        ],
        "volume_number": [
            (r"\b(?:volume(?:\s*no\.?|number)?|vol(?:\.|ume)?(?:\s*no\.?)?|খণ্ড)\s*[:\-]?\s*([A-Za-z\-]*[0-9০-৯][A-Za-z0-9০-৯\-]*)", True, 0.95),
        ],
        "deed_number": [
            (r"\b(?:deed\s*(?:no|number)|দলিল\s*(?:নং|নম্বর)?)\s*[:\-]?\s*([0-9০-৯A-Za-z\-\/]+)", True, 0.97),
            (r"\b(?:being\s*no\.?)\s*[:\-]?\s*([0-9০-৯]{2,})", True, 0.9),
            (r"\b(?:no|নং)\s*[:\-]?\s*([0-9০-৯]{3,})", True, 0.5),
        ],
        "transaction_type_major": [
            (r"\b(?:transaction\s*type\s*\(?major\)?|major\s*head)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.88),
        ],
        "transaction_type_minor": [
            (r"\b(?:transaction\s*type\s*\(?minor\)?|minor\s*head)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.88),
        ],
        "page_from": [
            (r"\b(?:page\s*from|starting\s*no\.?)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.9),
        ],
        "page_to": [
            (r"\b(?:page\s*to|end\s*no\.?)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.9),
        ],
        "additional_pages": [
            (r"\b(?:additional\s*pages?|অতিরিক্ত\s*পৃষ্ঠা)\s*[:\-]?\s*([0-9০-৯]+)", True, 0.87),
        ],
        "document_type": [
            (r"\b(?:document\s*type)\s*[:\-]?\s*([A-Za-z\u0980-\u09FF ,\-/]+)", False, 0.85),
            (r"\b(xerox|hand\s*written|handwritten|typed|টাইপ|হাতের\s*লেখা)\b", False, 0.7),
        ],
        "consideration_amount": [
            (r"\b(?:consideration\s*amount|transacted\s*amount|মূল্য|টাকা|amount)\s*[:\-]?\s*(?:rs\.?|inr|টাকা)?\s*([0-9০-৯,\.]+)", True, 0.9),
        ],
    }


FIELD_PATTERNS = build_field_patterns()

FIELD_REGION_PRIORITY = {
    "district": ["header", "left_margin", "signature", "stamp", "full_page"],
    "registration_office": ["header", "left_margin", "signature", "stamp", "full_page"],
    "book_type": ["header", "left_margin", "signature", "stamp", "full_page"],
    "year": ["header", "left_margin", "signature", "stamp", "full_page"],
    "volume_number": ["header", "left_margin", "signature", "stamp", "full_page"],
    "deed_number": ["left_margin", "header", "signature", "stamp", "full_page"],
    "transaction_type_major": ["header", "left_margin", "signature", "stamp", "full_page"],
    "transaction_type_minor": ["header", "left_margin", "signature", "stamp", "full_page"],
    "page_from": ["header", "left_margin", "signature", "stamp", "full_page"],
    "page_to": ["header", "left_margin", "signature", "stamp", "full_page"],
    "additional_pages": ["header", "left_margin", "signature", "stamp", "full_page"],
    "document_type": ["header", "left_margin", "signature", "stamp", "full_page"],
    "consideration_amount": ["left_margin", "header", "signature", "stamp", "full_page"],
}


def clip_confidence(value):
    return round(max(0.0, min(1.0, value)), 2)


def infer_confidence(base_score, region, extracted_value):
    region_boost = REGION_WEIGHT.get(region, 0.6)
    quality_boost = 0.0
    if extracted_value:
        l = len(str(extracted_value).strip())
        if 2 <= l <= 30:
            quality_boost += 0.05
        if re.search(r"[A-Za-z\u0980-\u09FF]", str(extracted_value)):
            quality_boost += 0.02
    return clip_confidence((base_score * region_boost) + quality_boost)


def validate_field(field, value):
    if value is None:
        return None, "missing"

    value = str(value).strip()
    if field in {"year", "volume_number", "deed_number", "page_from", "page_to", "additional_pages", "consideration_amount"}:
        value = normalize_bengali_digits(value)

    if field == "book_type":
        value = value.upper()
        return (value, "valid") if value in BOOK_TYPES else (value, "invalid")

    if field == "year":
        if re.fullmatch(r"(18|19|20)\d{2}", value):
            return value, "valid"
        return value, "invalid"

    if field in {"volume_number", "page_from", "page_to", "additional_pages"}:
        return (value, "valid") if re.fullmatch(r"\d+", value) else (value, "invalid")

    if field == "deed_number":
        return (value, "valid") if re.fullmatch(r"[A-Za-z0-9\-/]{2,}", value) else (value, "invalid")

    if field == "consideration_amount":
        value = value.replace(",", "")
        return (value, "valid") if re.fullmatch(r"\d+(\.\d+)?", value) else (value, "invalid")

    if field == "registration_office":
        value = normalize_registration_office(value)
        return (value, "valid") if len(value) >= 3 else (value, "invalid")

    return value, "valid" if value else "invalid"


def make_candidate(field, value, original_value, confidence, source_page, source_region, evidence, method):
    normalized_value, validation_status = validate_field(field, value)
    if validation_status == "invalid":
        confidence = min(confidence, 0.35)
    return {
        "field": field,
        "value": normalized_value,
        "original_value": original_value,
        "confidence": clip_confidence(confidence),
        "source_page": source_page,
        "source_region": source_region,
        "evidence_text": evidence,
        "method": method,
        "validation_status": validation_status,
    }


def generate_field_candidates(ocr_texts, page_number):
    candidates = {field: [] for field in CORE_FIELDS}

    page_range_pattern = re.compile(
        r"\b(?:pages?|পৃষ্ঠা)\s*([0-9০-৯]+)\s*(?:to|\-)\s*([0-9০-৯]+)",
        flags=re.I,
    )
    for region in FIELD_REGION_PRIORITY["page_from"]:
        text = ocr_texts.get(region)
        if not text:
            continue
        m = page_range_pattern.search(text)
        if not m:
            continue
        evidence = m.group(0).strip()
        candidates["page_from"].append(make_candidate(
            "page_from", m.group(1), m.group(1), infer_confidence(0.96, region, m.group(1)),
            page_number, region, evidence, "page_range_regex",
        ))
        candidates["page_to"].append(make_candidate(
            "page_to", m.group(2), m.group(2), infer_confidence(0.96, region, m.group(2)),
            page_number, region, evidence, "page_range_regex",
        ))
        break

    for field in CORE_FIELDS:
        for region in FIELD_REGION_PRIORITY[field]:
            text = ocr_texts.get(region)
            if not text:
                continue
            for pattern, is_numeric, base_score in FIELD_PATTERNS[field]:
                m = re.search(pattern, text, flags=re.I)
                if not m:
                    continue
                original_value = m.group(1).strip() if m.lastindex else m.group(0).strip()
                value = normalize_bengali_digits(original_value) if is_numeric else original_value
                confidence = infer_confidence(base_score, region, value)
                candidates[field].append(make_candidate(
                    field, value, original_value, confidence, page_number, region, m.group(0).strip(), "regex",
                ))

    return candidates


def select_best_candidate(candidates):
    valid = [c for c in candidates if c["validation_status"] == "valid"]
    pool = valid if valid else candidates
    if not pool:
        return None
    return max(pool, key=lambda c: (c["validation_status"] == "valid", c["confidence"]))


def select_page_fields(candidates):
    fields = {field: empty_field_payload() for field in CORE_FIELDS}
    for field, field_candidates in candidates.items():
        best = select_best_candidate(field_candidates)
        if best:
            fields[field] = {k: v for k, v in best.items() if k != "field"}
    return fields


def merge_document_fields(page_records):
    merged = {field: empty_field_payload() for field in CORE_FIELDS}
    all_candidates = {field: [] for field in CORE_FIELDS}

    for page in page_records:
        for field, candidates in page["field_candidates"].items():
            all_candidates[field].extend(candidates)

    for field, candidates in all_candidates.items():
        best = select_best_candidate(candidates)
        if best:
            merged[field] = {k: v for k, v in best.items() if k != "field"}

    validate_page_range(merged)
    return merged


def validate_page_range(fields):
    p_from = fields["page_from"]["value"]
    p_to = fields["page_to"]["value"]
    if p_from is None or p_to is None:
        return
    try:
        if int(p_from) > int(p_to):
            fields["page_from"]["validation_status"] = "invalid"
            fields["page_to"]["validation_status"] = "invalid"
            fields["page_from"]["confidence"] = min(fields["page_from"]["confidence"], 0.35)
            fields["page_to"]["confidence"] = min(fields["page_to"]["confidence"], 0.35)
    except ValueError:
        pass


def to_legacy_flat_fields(structured_fields):
    return {field: structured_fields.get(field, {}).get("value") for field in CORE_FIELDS}


# =========================
# PAGE / DOCUMENT PROCESSING
# =========================
def run_ocr_for_page(page_image_path, crop_dir, output_root, doc_id, page_number, include_full_page_ocr=False):
    ocr_texts = {}

    if include_full_page_ocr:
        ocr_texts["full_page"] = ocr_image(page_image_path, output_root, doc_id, page_number, "full_page")

    for region in OCR_REGIONS:
        path = os.path.join(crop_dir, f"{region}.png")
        if not os.path.exists(path):
            print(f"      [INFO] Region not found, skipping OCR: {region}")
            continue
        ocr_texts[region] = ocr_image(path, output_root, doc_id, page_number, region)
        print(f"      [INFO] OCR ready for {region} (chars={len(ocr_texts[region])})")

    return ocr_texts


def classify_document_type(crop_dir, output_root, doc_id, page_number):
    votes = []
    evidence_bits = []

    for region in DOC_TYPE_REGIONS:
        path = os.path.join(crop_dir, f"{region}.png")
        if not os.path.exists(path):
            continue
        payload = classify_image_style(path, output_root, doc_id, page_number, region)
        votes.append((region, payload["label"], payload["raw_label"]))
        evidence_bits.append(f"{region}:{payload['raw_label']}")

    if not votes:
        return None

    counts = {"printed": 0, "handwritten": 0, "mixed": 0}
    for _, label, _ in votes:
        counts[label] += 1

    chosen = max(counts, key=counts.get)
    total = len(votes)
    confidence = counts[chosen] / total
    if counts["printed"] > 0 and counts["handwritten"] > 0:
        chosen = "mixed"
        confidence = max(confidence, (counts["printed"] + counts["handwritten"]) / total)

    return make_candidate(
        "document_type", chosen, chosen, confidence, page_number,
        votes[0][0] if len(votes) == 1 else "multi_region",
        " | ".join(evidence_bits), "style_classifier",
    )


def process_page(document, page, output_root, include_full_page_ocr=False):
    doc_id = document["doc_id"]
    page_number = page["page_number"]
    image_path = page["image_path"]
    print(f"   [PAGE] {page_number}: {os.path.basename(image_path)}")

    detections = detect_regions(image_path)
    print(f"      [INFO] YOLO detections: {len(detections)}")

    crop_dir = crop_and_save(image_path, detections, output_root, doc_id, page_number)
    if not crop_dir:
        return None

    ocr_texts = run_ocr_for_page(
        image_path,
        crop_dir,
        output_root,
        doc_id,
        page_number,
        include_full_page_ocr=include_full_page_ocr,
    )
    field_candidates = generate_field_candidates(ocr_texts, page_number)
    doc_type_candidate = classify_document_type(crop_dir, output_root, doc_id, page_number)
    if doc_type_candidate:
        field_candidates["document_type"].append(doc_type_candidate)

    selected_fields = select_page_fields(field_candidates)
    resolved = [field for field, payload in selected_fields.items() if payload["value"] is not None]
    print(f"      [INFO] Page extracted fields: {resolved if resolved else 'none'}")

    return {
        "page_number": page_number,
        "image_path": image_path,
        "crop_dir": crop_dir,
        "detections": detections,
        "ocr_texts": ocr_texts,
        "field_candidates": field_candidates,
        "selected_fields": selected_fields,
    }


def process_document(document, output_root, legacy_flat_output=False, include_full_page_ocr=False):
    print(f"\n[DOC] Processing: {document['source_path']}")
    page_records = []

    for page in document["pages"]:
        try:
            page_record = process_page(
                document,
                page,
                output_root,
                include_full_page_ocr=include_full_page_ocr,
            )
            if page_record:
                page_records.append(page_record)
        except Exception as exc:
            print(f"   [ERROR] Failed page {page['page_number']}: {exc}")

    document_fields = merge_document_fields(page_records)
    resolved = [field for field, payload in document_fields.items() if payload["value"] is not None]
    unresolved = [field for field in CORE_FIELDS if field not in resolved]
    print(f"   [INFO] Document fields: {resolved if resolved else 'none'}")
    print(f"   [INFO] Unresolved document fields: {unresolved if unresolved else 'none'}")

    if legacy_flat_output:
        document_fields = to_legacy_flat_fields(document_fields)
        for page in page_records:
            page["selected_fields"] = to_legacy_flat_fields(page["selected_fields"])

    return {
        "doc_id": document["doc_id"],
        "source_path": document["source_path"],
        "source_type": document["source_type"],
        "document_fields": document_fields,
        "pages": page_records,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO + PaddleOCR-VL deed extraction pipeline")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory containing page images and/or PDFs.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for crops, cache, and results.")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="DPI for PDF page rendering when PDFs are present.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N documents.")
    parser.add_argument("--legacy-flat-output", action="store_true", help="Emit flat selected field values.")
    parser.add_argument(
        "--full-page-ocr",
        action="store_true",
        help="Also OCR full pages for fallback candidates. Slower and noisier; disabled by default.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    documents = discover_documents(args.input_dir, args.output_dir, args.pdf_dpi)
    if not documents:
        print(f"[ERROR] No images or PDFs found in: {os.path.abspath(args.input_dir)}")
        return

    print(f"[INFO] Found {len(documents)} document input(s) in {args.input_dir}")
    if args.limit is not None:
        if args.limit <= 0:
            print(f"[ERROR] --limit must be > 0, got: {args.limit}")
            return
        documents = documents[:args.limit]
        print(f"[INFO] Applying limit: processing first {len(documents)} document(s)")

    all_results = []
    for document in documents:
        all_results.append(process_document(
            document,
            args.output_dir,
            legacy_flat_output=args.legacy_flat_output,
            include_full_page_ocr=args.full_page_ocr,
        ))

    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Completed. Processed={len(all_results)}/{len(documents)}")
    print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
