# Model Card — Heart Risk Classifier

## Model Details
- **Type:** RandomForestClassifier (sklearn Pipeline)
- **Version:** See MLflow Model Registry
- **Framework:** scikit-learn 1.7.2

## Intended Use
Screening tool to flag patients for further clinical evaluation.
**Not** intended as a standalone diagnostic device.

## Dataset
- **Source:** UCI Heart Disease Dataset (Cleveland)
- **Size:** 303 patients, 13 features
- **Target:** Binary — 0 (no disease), 1 (disease present)

## Performance (best run)
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9491 |
| Accuracy | 0.8689 |
| F1 | 0.8621 |
| Recall | 0.8929 |
| Precision | 0.8333 |

## Limitations
- Dataset is small (303 records) and from a single institution
- May not generalize to populations outside the study demographic
- Missing value imputation may introduce bias in `ca` and `thal` features

## Ethical Considerations
- False negatives (missed disease) are clinically more dangerous than false positives
- Recall is the primary operational metric for deployment decisions
- Model should always be used alongside clinical judgment