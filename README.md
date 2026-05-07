# Land Deed Extraction

Local OCR and field extraction experiments for scanned handwritten and printed land deed documents.

The current pipeline combines a trained YOLO detector with PaddleOCR-VL 1.5 to crop document regions, OCR the useful non-body regions, and extract deed metadata into JSON.

## Current Pipeline

`ocr_pipeline.py` currently:

1. Loads YOLO weights from `runs/detect/train-main/weights/best.pt`.
2. Detects page regions such as `left_margin`, `header`, `signature`, `stamp`, and `main_body`.
3. Crops detected regions into `pipeline_output/<document_id>/`.
4. OCRs selected non-`main_body` regions with `PaddlePaddle/PaddleOCR-VL-1.5`.
5. Extracts structured fields with source evidence and heuristic confidence.

Current core fields:

- `district`
- `registration_office`
- `book_type`
- `year`
- `volume_number`
- `deed_number`
- `transaction_type_major`
- `transaction_type_minor`
- `page_from`
- `page_to`
- `additional_pages`
- `document_type`
- `consideration_amount`

## Setup

This project uses `uv` for running inside the local Python environment.

Expected key dependencies:

- `torch`
- `transformers` v5
- `ultralytics`
- `opencv-python`
- `Pillow`
- `einops`

Run the current validation pipeline:

```bash
uv run python ocr_pipeline.py --limit 5
```

Run with the older scalar field shape:

```bash
uv run python ocr_pipeline.py --limit 5 --legacy-flat-output
```

## Output

The pipeline writes:

```text
pipeline_output/results.json
```

Each extracted field uses a structured payload:

```json
{
  "value": "796",
  "original_value": "796",
  "confidence": 0.91,
  "source_region": "left_margin",
  "evidence_text": "BEING NO. 796"
}
```

Missing fields are represented with `null` values and `confidence: 0.0`.

## Known Limitations

- The current image-mode pipeline processes page images, not source PDFs.
- Regex-only extraction does not scale across all page layouts.
- Some VLM classification prompts may return OCR text instead of a clean class label, so the pipeline uses pragmatic fallbacks.
- `main_body` handwritten Bengali extraction is intentionally deferred until the non-body metadata extraction is reliable.
- `pipeline_output/`, input image folders, training datasets, and most YOLO run artifacts are treated as generated/local data and are not committed.

## Planned Scalable Architecture

The next pipeline version should move to a staged, evidence-first design:

1. Accept PDFs as primary input and render pages to images.
2. Detect page regions with YOLO.
3. Cache OCR text for full pages and crops.
4. Classify page and region types before extraction.
5. Generate multiple candidates per field from OCR text, targeted prompts, and patterns.
6. Validate candidates with field-specific rules.
7. Merge page-level candidates into final PDF-level document fields.

The goal is to output both page-level debug records and final document-level JSON with source page, source region, confidence, validation status, and evidence text for each selected field.
