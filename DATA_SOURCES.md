# Where the data comes from

The raw datasets are not redistributed here. Download them from the sources below
into `data/raw/`, then run `python src/harmonize.py`.

## UCI Heart Disease — Cleveland, Hungarian, VA Long Beach

https://archive.ics.uci.edu/dataset/45/heart+disease
DOI 10.24432/C52P4X

Files: `processed.cleveland.data`, `processed.hungarian.data`,
`processed.va.data`. The outcome is angiographically documented coronary disease,
binarised as any narrowing greater than 50% against none.

Statlog Heart is deliberately NOT used: all 270 of its records duplicate Cleveland
records. See the overlap audit in `src/harmonize.py`.

## Framingham teaching extract

A public educational extract of 4,240 participants with a ten-year incident
coronary heart disease indicator. It is not the Framingham Heart Study research
release and not the sample analysed in D'Agostino et al., Circulation 2008, from
which it derives only indirectly. Access to the research release is through BioLINCC.

## Kaggle cardiovascular disease dataset

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Ulianova S, 2019. 70,000 records; the outcome is disease status at examination.
The collection protocol is not described in any peer-reviewed source, which the
manuscript states as a limitation.

Cleaning removes 1,402 records (2.0%), leaving 68,598. The rules and the count
removed by each are printed by `src/harmonize.py`.
