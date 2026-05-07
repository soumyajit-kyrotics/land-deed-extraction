# Land Deed Extraction

Local OCR and field extraction experiments for scanned handwritten and printed land deed documents.

The current pipeline combines a trained YOLO detector with PaddleOCR-VL 1.5 to crop document regions, OCR the useful non-body regions, and extract deed metadata into JSON.

## Current Pipeline

`ocr_pipeline.py` currently:

1. Loads YOLO weights from `runs/detect/train-main/weights/best.pt`.
2. Accepts page images and, when PyMuPDF is installed, PDFs.
3. Detects page regions such as `left_margin`, `header`, `signature`, `stamp`, and `main_body`.
4. Crops detected regions into `pipeline_output/crops/<document_id>/page_XXXX/`.
5. OCRs selected non-`main_body` regions with `PaddlePaddle/PaddleOCR-VL-1.5`.
6. Caches OCR/style outputs under `pipeline_output/cache/`.
7. Generates field candidates, validates them, and merges page-level evidence into document-level fields.

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
- `pymupdf` for PDF input mode

Run the current validation pipeline:

```bash
uv run python ocr_pipeline.py --limit 5
```

Run against a custom input directory:

```bash
uv run python ocr_pipeline.py --input-dir path/to/documents --limit 5
```

Enable full-page OCR fallback candidates when needed:

```bash
uv run python ocr_pipeline.py --limit 5 --full-page-ocr
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
  "source_page": 1,
  "source_region": "left_margin",
  "evidence_text": "BEING NO. 796",
  "method": "regex",
  "validation_status": "valid"
}
```

Missing fields are represented with `null` values and `confidence: 0.0`.

## Known Limitations

- PDF mode requires `pymupdf`; image mode works without it.
- Full-page OCR is available behind `--full-page-ocr`, but crop-level OCR is the default because full-page generations can be slower and noisier.
- Regex extraction is now candidate-based, but field coverage still needs tuning on a reviewed seed set.
- Some VLM classification prompts may return OCR text instead of a clean class label, so the pipeline uses pragmatic fallbacks.
- `main_body` handwritten Bengali extraction is intentionally deferred until the non-body metadata extraction is reliable.
- `pipeline_output/`, input image folders, training datasets, and most YOLO run artifacts are treated as generated/local data and are not committed.

## Planned Scalable Architecture

The next pipeline version should deepen the staged, evidence-first design:

1. Add a reviewed seed set for 50-100 PDFs/pages.
2. Tune page/region classifiers against the seed set.
3. Add registration office alias dictionaries and district-level validators.
4. Add targeted prompts only for low-confidence fields.
5. Improve PDF-level merge rules with cross-page consistency checks.

The goal is to output both page-level debug records and final document-level JSON with source page, source region, confidence, validation status, and evidence text for each selected field.
