# Project Brief: Photo-Robust Floorplan Segmentation

## Short Description

This project adapts a CubiCasa5K floorplan segmentation model to work on photographed floorplans instead of only clean digital floorplans.

Clean floorplan images are usually high quality and already aligned. Real photographs are harder: they contain perspective distortion, shadows, uneven lighting, blur, cropping, and compression. The project builds a photographed floorplan pipeline, fine-tunes the CubiCasa model on that domain, and evaluates whether the predicted walls and doors still form a usable floorplan structure.

## Problem

Most floorplan segmentation models are tested on clean raster floorplans. In practice, a user may take a phone photo of a printed or displayed floorplan. That photo can damage the segmentation output even if the original floorplan is simple.

The key issue is not only pixel accuracy. A wall mask can look partly correct but still be structurally unusable if the walls become fragmented or rooms are no longer enclosed.

## What I Built

- A preprocessing pipeline that rectifies and crops photographed floorplans.
- A fine-tuning dataset pipeline using CubiCasa-derived masks.
- A fine-tuned CubiCasa segmentation model for photographed inputs.
- A paired clean-vs-photo evaluation setup.
- Structural validity metrics for measuring floorplan usability.
- Experiments for baseline comparison, degradation sensitivity, homography/crop effects, and simple preprocessing mitigation.
- Scripts for generating thesis tables and appendix figures.

## Technical Pipeline

```text
Raw photo
  -> page rectification and floorplan crop
  -> aligned photo benchmark
  -> fine-tuned CubiCasa segmentation model
  -> wall and door masks
  -> structural validity checks
  -> result tables and failure analysis
```

## Model

The project uses the CubiCasa/FloorplanTransformation architecture as the base model. The main fine-tuning script is:

```text
src/training/01_finetune_cubicasa_photos.py
```

The fine-tuned output focuses on wall-like and door predictions because those are the most important structures for downstream floorplan interpretation.

## Evaluation Idea

The evaluation compares the same plan under two conditions:

- clean raster input
- photographed input

This makes it possible to identify cases where the model works on the clean version but fails on the photographed version.

The main paired evaluation script is:

```text
src/experiments/00_paired_clean_vs_photo_evaluation.py
```

## Structural Metrics

The project uses structural indicators such as:

- wall connected components: high values suggest fragmented walls
- enclosure count: whether the prediction forms enclosed room-like regions
- enclosed free-space ratio: how much free space is inside enclosed regions
- outside free-space ratio: how much free space leaks to the outside

These metrics are used because floorplans need to be structurally coherent, not just visually similar.

## Key Experiments

| Experiment | Purpose |
|---|---|
| Clean vs photographed | Measures how much photo capture degrades the same plan |
| Baseline vs fine-tuned | Shows whether photo-domain fine-tuning improves robustness |
| Factor ladder sensitivity | Tests which distortions cause structural failure |
| Homography and crop analysis | Checks whether geometric correction explains failures |
| Mitigation test | Tests simple preprocessing such as contrast enhancement |
| Rule ablation | Checks how sensitive the validity decision is to threshold choices |

## How To Explain It Quickly

The shortest explanation is:

> I worked on making floorplan segmentation more robust when the input is a real photograph instead of a clean digital image. I built a photo preprocessing and fine-tuning pipeline for CubiCasa5K, trained the model on photographed floorplans, and evaluated the results with structural metrics that check whether walls remain connected and rooms remain enclosed.

## What To Show Instead Of A Live Demo

If the dataset or GPU environment is unavailable, show static outputs in this order:

1. raw photographed floorplan
2. rectified/cropped benchmark image
3. predicted wall and door masks
4. clean-vs-photo overlay
5. `paired_results.csv` summary
6. mitigation figure
7. final thesis table

This demonstrates the full pipeline without needing to run the model live.

