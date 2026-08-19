# GeoRescue AI Experiment Protocol

## Objective

Measure whether multimodal optical + SAR fusion improves pixel-level disaster-damage segmentation and cross-event generalization versus single-modality baselines.

## Controlled comparisons

All baseline experiments should use the same:

- train/validation/test event split;
- preprocessing and spatial resolution;
- loss function;
- optimizer and training budget;
- augmentation policy;
- evaluation metrics;
- checkpoint selection rule.

Only the input modality or fusion architecture should change when an ablation is intended to isolate a component.

## Required reporting

For each experiment record:

- model name and version;
- dataset and exact version/date;
- event split;
- channels and spatial resolution;
- number of training/validation/test samples;
- class distribution;
- seed;
- batch size;
- optimizer and learning rate;
- epochs and early-stopping policy if used;
- best checkpoint criterion;
- mIoU;
- Dice/F1;
- precision;
- recall;
- per-class IoU;
- confusion matrix;
- qualitative prediction examples;
- known failure modes.

## Anti-leakage rule

Never split individual patches randomly across train and test when the research question concerns unseen-disaster-event generalization. Event identity must remain separated between training and evaluation for that experiment.
