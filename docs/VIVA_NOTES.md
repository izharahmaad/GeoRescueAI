# Viva Notes

## One-line project definition

GeoRescue AI is a multimodal deep-learning pipeline that combines optical and SAR satellite imagery to produce pixel-level disaster-damage maps.

## Core research question

Does multimodal optical + SAR fusion improve segmentation quality and generalization to unseen disaster events compared with optical-only and SAR-only baselines?

## Why SAR?

SAR is an active microwave sensing modality that can provide structural/surface information under conditions where optical imagery is limited by daylight and cloud cover.

## Why segmentation?

The task needs a spatial damage map, so the model should predict a class for each pixel rather than a single label for the whole image.

## Why U-Net?

U-Net provides a strong, interpretable encoder-decoder segmentation baseline with skip connections that help preserve spatial detail.

## Why compare three models?

Optical-only and SAR-only baselines establish modality-specific performance. Fusion tests whether complementary information improves the task.

## What would make the research claim credible?

The result must come from controlled experiments, repeated or stable runs where appropriate, a leakage-safe event split, and transparent reporting of both improvements and failure cases.
