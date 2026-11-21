# Kidney Ultrasound APN Research Code

Analysis scripts for research on acute pyelonephritis (APN) classification from
pediatric renal ultrasound, using kidney contours and DMSA-referenced labels.

**Research use only.** This code is not a clinical diagnostic device.

## Contents

`05_code/root_files/` holds the analysis scripts: cohort audits, feature
extraction, frozen-encoder probes, model fitting and evaluation, segmentation
fine-tuning, and unit tests (`test_*.py`). `05_code/artifact_builders/` holds a
blank data-collection form builder.

Scripts were written for the authors' local project layout and refer to paths
such as `02_experiments/` and `03_data_models/`, which are not part of this
repository.

## Not included

- Patient images, contours, clinical workbooks, labels, and identifiers
- Experiment outputs, trained weights, and model checkpoints
- Manuscripts, manuscript build scripts, and figure scripts

Hospital chart numbers that appeared in two scripts have been replaced with
placeholders. Running the scripts requires the original data, which are not
publicly available.

## Environment

Python 3.11 with numpy, pandas, scikit-learn, PyTorch, torchvision, OpenCV and
Pillow.
