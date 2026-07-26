# Model Card

## Model overview

This repository contains two supervised classification benchmarks trained on a
historical public corporate-rating dataset:

- a binary investment-grade versus speculative-grade classifier;
- a seven-band ordered rating classifier.

The binary benchmark uses class-weighted logistic regression. The seven-band
extension uses a class-weighted random forest. These models were selected on
2015 validation data and evaluated once on a 2016 chronological holdout.

## Intended use

The models are intended to demonstrate:

- leakage-aware feature and target design;
- temporal and issuer-grouped validation;
- reproducible preprocessing and evaluation;
- communication of model limitations.

They may be used as an educational benchmark or portfolio demonstration.

## Out-of-scope use

The models must not be used to:

- assign current credit ratings;
- make lending, underwriting, pricing, or investment decisions;
- replace agency methodology or analyst judgment;
- infer the creditworthiness of companies outside this historical dataset.

## Data

The source contains 7,805 rating observations from 2010–2016. Predictors include
16 financial ratios, rating agency, sector, and SIC code.

The following fields are explicitly excluded from modeling:

- `Rating` and both constructed targets;
- source `Binary Rating`, because it is derived from the target;
- corporation name, CIK, and ticker;
- rating date and derived year.

CIK is used only to construct issuer-grouped validation folds.

## Validation

### Model selection

- Training period: 2010–2014
- Validation period: 2015
- Selection metric:
  - balanced accuracy for the binary task;
  - macro-F1 for the seven-band task.

### Final evaluation

- Development period after selection: 2010–2015
- Untouched chronological holdout: 2016
- Robustness check: five-fold stratified group validation by CIK

## Verified results

| Task | Model | 2016 balanced accuracy | 2016 macro-F1 |
|---|---|---:|---:|
| Binary | Logistic regression | 0.788 | 0.777 |
| Seven-band | Random forest | 0.451 | 0.405 |

Additional binary results:

- speculative-grade recall: 0.877;
- ROC-AUC: 0.881;
- average precision: 0.836;
- issuer-grouped balanced accuracy: 0.704 ± 0.040.

Additional seven-band results:

- ordinal mean absolute error: 0.830 rating bands;
- predictions within one band: 82.8%;
- issuer-grouped macro-F1: 0.353 ± 0.047.

## Key limitations

- The data predates current market conditions and rating methodologies.
- The public ratios omit qualitative, structural, macroeconomic, and
  forward-looking information.
- Multiple agencies may apply different methodologies.
- Identical predictor vectors can map to different ratings, indicating label
  ambiguity not explained by the available variables.
- Issuer-grouped performance is lower than chronological performance,
  especially for the seven-band task.
- The dataset does not support a credible headline benchmark across all 23
  original rating notches.

## Monitoring and governance

This is a static benchmark, so there is no production monitoring. Any future
deployment would require, at minimum:

- refreshed and licensed data;
- documented agency-scale harmonization;
- drift, calibration, and subgroup monitoring;
- model approval, independent validation, and change controls;
- human review and a clear override process.
