# Two-Stage LLM Pipeline for Brick Mapping from BMS Screenshots

This repository contains the code, prompts and selected experimental artifacts for a two-stage large language model pipeline for Brick class identification from Building Management System (BMS) screenshots.

## Overview

The pipeline separates screenshot interpretation from semantic reasoning into two sequential stages:

1. **Stage 1: Structured Visual Extraction**  
   A vision-capable model extracts visible point identifiers, values, units and related visual cues from BMS screenshots and returns them in a normalized JSON format.

2. **Stage 2: Ontology-Grounded Semantic Reasoning**  
   A reasoning-focused model maps the extracted point records to Brick-compatible classes. This stage can be run either:
   - **without retrieval augmentation**, or
   - **with retrieval augmentation (RAG)** using curated Brick ontology documentation.

This staged design makes the pipeline more auditable and easier to evaluate, since visual extraction errors can be separated from semantic classification errors.

## Repository Contents

### Core scripts
- `benchmark_ocr_vision.py`  
  Runs Stage 1 structured visual extraction over a directory of screenshots.
- `benchmark_stage2_reasoning.py`  
  Runs Stage 2 semantic reasoning without retrieval augmentation.
- `benchmark_stage2_reasoning_rag.py`  
  Runs Stage 2 semantic reasoning with retrieval augmentation.
- `compare_ocr_outputs.py`  
  Evaluates Stage 1 outputs under stricter schema-level matching.
- `compare_ocr_outputs_relaxed.py`  
  Evaluates Stage 1 outputs under relaxed matching focused on downstream utility.
- `compare_stage2_outputs.py`  
  Evaluates Stage 2 predicted mappings against annotated reference labels.
- `make_ocr_gold_template.py`  
  Creates template files for manual OCR ground-truth annotation.
- `make_stage2_inputs.py`  
  Converts OCR outputs into the input format used by Stage 2.

### Retrieval scripts
- `rag.py`  
  Retrieval helper used in the final reported RAG pipeline.
- `ingest_brick_v31.py`  
  Brick ingestion script used to build the final retrieval collection used in the paper.

### Prompt files
- `prompts/stage1/bms_ocr_prompt.txt`  
  Final Stage 1 OCR prompt.
- `prompts/stage1/bms_ocr_prompt_short.txt`  
  Earlier shorter OCR prompt retained for comparison.
- `prompts/stage2/bms_stage2_prompt.txt`  
  Final Stage 2 prompt without retrieval.
- `prompts/stage2/bms_stage2_prompt_rag.txt`  
  Final Stage 2 prompt with retrieval augmentation.

### Data folders
- `bms_ocr_eval/`  
  Stage 1 evaluation assets.
- `bms_stage2_eval/`  
  Stage 2 evaluation assets.

### Other folders
- `Brick/`, `data/`  
  Local ontology/documentation sources used during retrieval development.
- `chroma_db/`  
  Local Chroma vector store generated during retrieval indexing.

## Final Reported Setup

The paper reports results for the following final setup:

- **Stage 1 model:** `Gemma 3 27B`
- **Stage 2 model:** `Ministral 3 14B Reasoning`
- **Stage 2 retrieval condition:** optional RAG over curated Brick ontology material
- **Final ingestion script:** `ingest_brick_v31.py`
- **Final retrieval helper:** `rag.py`

All experiments were executed programmatically through Python scripts rather than through an interactive chat interface, to ensure consistent inference settings and output collection.

## Environment

The scripts assume a locally served OpenAI-compatible LM Studio endpoint. A common default configuration is:

    export LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
    export LMSTUDIO_API_KEY="lm-studio"
    export LMSTUDIO_MODEL="gemma-3-27b-it"

For Stage 2, set the model variable to the reasoning model used in the paper:

    export LMSTUDIO_MODEL="ministral-3-14b-reasoning"

The `LMSTUDIO_BASE_URL` and `LMSTUDIO_API_KEY` variables affect execution and should be set explicitly. If your LM Studio server is running on a different host or port, replace `LMSTUDIO_BASE_URL` with the endpoint used in your local setup. The value `lm-studio` is the standard local placeholder key used by LM Studio and is not a secret. If your local setup uses a different exact Stage 2 model string, replace the example above with the model name exposed by your LM Studio installation.

## Expected Workflow

### 1. Stage 1: OCR / Structured Visual Extraction

Run Stage 1 over a set of screenshots:

    python benchmark_ocr_vision.py \
      --input-dir "./bms_ocr_eval/test_images" \
      --output "./bms_ocr_eval/test_outputs_gemma27b.jsonl" \
      --prompt-file "./prompts/stage1/bms_ocr_prompt.txt" \
      --raw-dir "./bms_ocr_eval/test_raw_outputs_gemma27b" \
      --max-tokens 6000

### 2. Stage 1 Evaluation

Strict evaluation:

    python compare_ocr_outputs.py \
      --predictions "./bms_ocr_eval/test_outputs_gemma27b.jsonl" \
      --gold-dir "./bms_ocr_eval/test_gold" \
      --output "./bms_ocr_eval/test_comparison.json"

Relaxed evaluation:

    python compare_ocr_outputs_relaxed.py \
      --predictions "./bms_ocr_eval/test_outputs_gemma27b.jsonl" \
      --gold-dir "./bms_ocr_eval/test_gold" \
      --output "./bms_ocr_eval/test_comparison_relaxed.json"

### 3. Prepare Stage 2 Inputs

Convert Stage 1 outputs into Stage 2 input format:

    python make_stage2_inputs.py \
      --input "./bms_ocr_eval/test_outputs_gemma27b.jsonl" \
      --output "./bms_stage2_eval/test_stage2_inputs.jsonl"

### 4. Stage 2 Without RAG

    python benchmark_stage2_reasoning.py \
      --input "./bms_stage2_eval/test_stage2_inputs.jsonl" \
      --output "./bms_stage2_eval/test_stage2_outputs_no_rag.jsonl" \
      --prompt-file "./prompts/stage2/bms_stage2_prompt.txt" \
      --max-tokens 1800

### 5. Build the Retrieval Collection

    python ingest_brick_v31.py

### 6. Stage 2 With RAG

    python benchmark_stage2_reasoning_rag.py \
      --input "./bms_stage2_eval/test_stage2_inputs.jsonl" \
      --output "./bms_stage2_eval/test_stage2_outputs_rag.jsonl" \
      --prompt-file "./prompts/stage2/bms_stage2_prompt_rag.txt" \
      --max-tokens 1800 \
      --use-rag \
      --chroma-dir "./chroma_db" \
      --brick-collection "brick_stage2_task_docs_v31" \
      --top-k 2 \
      --max-retrieved-chars 900

### 7. Stage 2 Evaluation

    python compare_stage2_outputs.py \
      --predictions "./bms_stage2_eval/test_stage2_outputs_rag.jsonl" \
      --gold "./bms_stage2_eval/test_gold.jsonl" \
      --output "./bms_stage2_eval/test_stage2_comparison_rag.json"

## Evaluation Logic

### Stage 1

Stage 1 is evaluated using precision, recall, and F1 under:
- **strict matching**, based on the full extracted schema except `id`
- **relaxed matching**, based on the tuple:
  - `base_id`
  - `value`
  - `unit`

The relaxed metric better reflects extraction utility for downstream semantic reasoning.

### Stage 2

Stage 2 is evaluated using precision, recall, and F1 for point-level Brick-compatible class assignment under:
- **ground-truth OCR input**
- **generated OCR input**
- **without RAG**
- **with RAG**

This design makes it possible to distinguish:
- semantic reasoning quality under clean input
- error propagation from Stage 1
- the added value of retrieval augmentation

## Notes on Data and Reproducibility

This repository is intended as supplementary research material for the paper. Depending on sharing constraints, not all raw screenshots or annotations may be included in public form. Where raw assets are omitted, the repository still provides:
- the final prompts
- the evaluation scripts
- the ingestion and retrieval scripts
- the execution structure used for the reported experiments

## Citation

If you use this repository, please cite the associated paper.

## Contact

For questions regarding the code or supplementary material, please contact the repository owner.
