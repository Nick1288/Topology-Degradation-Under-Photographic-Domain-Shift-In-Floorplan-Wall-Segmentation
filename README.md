# Photo-Robust Floorplan Segmentation

Code for an FYP/thesis project on adapting CubiCasa5K floorplan segmentation to photographed floorplan inputs.

The project investigates how real-world photo capture affects floorplan segmentation. Clean digital floorplans are converted into a photographed-input workflow, the CubiCasa model is fine-tuned on that domain, and the outputs are evaluated using both prediction masks and structural validity metrics.

## Repository Contents

```text
src/
  common/              Shared utilities for model loading, preprocessing, and structural metrics
  data_preparation/    Photo rectification, SVG mask extraction, and dataset preparation
  training/            Fine-tuning code
  evaluation/          Fine-tuned model inference and prediction export
  experiments/         Clean-vs-photo, baseline-vs-finetuned, degradation, and mitigation studies
  reporting/           Scripts that generate result tables and appendix figures
  floortrans/          CubiCasa/FloorplanTransformation model dependency
docs/
  index.html           Static project page for GitHub Pages
  project_brief.md     Project summary and technical scope
  workflow.md          Pipeline diagram and script map
```

## Project Page

For a visual overview of the project, open:

```text
docs/index.html
```

This page presents the problem, pipeline, headline results, and key figures.

## Main Training Script

The fine-tuning script used for the thesis is:

```text
src/training/01_finetune_cubicasa_photos.py
```

This script fine-tunes the CubiCasa/FloorplanTransformation backbone for photographed floorplan inputs.

## Core Workflow

1. Build a rectified photo benchmark from raw photographed floorplans.
2. Prepare fine-tuning labels from CubiCasa SVG-derived masks.
3. Fine-tune the segmentation model.
4. Export wall-like and door predictions.
5. Compare clean raster inputs against photographed inputs.
6. Run degradation, mitigation, and reporting scripts.

## Important Scripts

| Stage | Script |
|---|---|
| Photo benchmark | `src/data_preparation/01_build_photo_benchmark.py` |
| Fine-tuning data | `src/data_preparation/02_prepare_finetuning_dataset.py` |
| Training | `src/training/01_finetune_cubicasa_photos.py` |
| Evaluation | `src/evaluation/01_evaluate_finetuned_model.py` |
| Prediction export | `src/evaluation/02_export_prediction_masks.py` |
| Clean-vs-photo evaluation | `src/experiments/00_paired_clean_vs_photo_evaluation.py` |
| Baseline-vs-finetuned comparison | `src/experiments/01_compare_base_vs_finetuned.py` |
| Degradation sensitivity | `src/experiments/03_factor_ladder_sensitivity.py` |
| Mitigation testing | `src/experiments/06_mitigation_test.py` |
| Thesis tables | `src/reporting/01_generate_thesis_tables.py` |

## Setup

Run scripts from the repository root with `src` on `PYTHONPATH`.

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
pip install -r requirements.txt
```

Bash:

```bash
export PYTHONPATH="$PWD/src"
pip install -r requirements.txt
```

## Example Commands

```powershell
python src\data_preparation\01_build_photo_benchmark.py `
  --photo-root path\to\raw_photos `
  --data-root path\to\cubicasa5k `
  --out-root path\to\photo_benchmark

python src\training\01_finetune_cubicasa_photos.py `
  --data-root path\to\prepared_finetune_data `
  --pretrained-backbone path\to\model_best_val_loss_var.pkl `
  --out-dir runs_finetune `
  --mode w3d

python src\evaluation\02_export_prediction_masks.py `
  --data-root path\to\photo_benchmark `
  --ckpt runs_finetune\best.pth `
  --out-root preds_photo

python src\experiments\00_paired_clean_vs_photo_evaluation.py `
  --photo-pred-root preds_photo `
  --clean-raster-root path\to\clean_rasterized `
  --ckpt runs_finetune\best.pth `
  --out-root paired_eval_out
```

## Evaluation Metrics

The project uses structural validity checks in addition to segmentation outputs:

- `wall_cc`: connected wall components, used as a wall fragmentation indicator.
- `enclosure_count`: number of enclosed free-space regions.
- `enclosed_free_ratio`: proportion of free space inside enclosed regions.
- `outside_free_ratio`: proportion of free space connected to the outside.

These metrics help identify cases where a mask may look acceptable locally but fails as a usable floorplan structure.

## Excluded Files

The repository does not include datasets, raw photographs, generated outputs, debug folders, or model checkpoints. Those files are large and should be stored separately.
