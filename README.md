# Chicago Crash Dashboard

**[Live app →](https://chicago-crash-dashboard.onrender.com)**

Which conditions are associated with injury in Chicago traffic crashes, and how much of that
association is available before the crash rather than reconstructed afterwards?

On 49,837 crashes reported to the Chicago Police Department (January 2021 to July 2026), logistic
regression reaches AUC 0.812 for any injury and 0.833 for severe injury but restricted to fields
known before impact, AUC falls to 0.665 and 0.649. Most of the prediction comes
from post-hoc report fields, not from road or weather conditions.

Three-page Dash app: a per-crash risk verdict, citywide crash patterns, and a crash map.

## Data

[Traffic Crashes – Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if),
Chicago Open Data Portal. 50,000-crash sample, 49,837 after dropping rows with a missing injury
field.

| Type of injury | Definition | Rate |
|---|---|---|
| `INJURY` | `INJURIES_TOTAL > 0` | 15.4% (7,696 crashes) |
| `SEVERE` | incapacitating or fatal | 1.7% (849 crashes) |

## Method

Logistic regression on 5 numeric and 6 categorical features, one-hot encoded inside the pipeline
so a single row at inference time gets the columns the model was trained on.

Split 60/20/20 into train, validation and test. Because both targets are rare, almost nothing
clears the default 0.5 cutoff, the decision threshold is swept on validation and applied once
to the held-out test set. The chosen thresholds differ by target (0.22 for injury, 0.12 for severe),
which is why a single shared threshold is not appropriate.

Accuracy is reported but is not the metric to read: a model that never predicts an injury scores
84.6% on the injury target and 98.3% on the severe target.

## Results

### Injury model

| | AUC-ROC | PR-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| All features | 0.812 | 0.534 | 0.434 | 0.545 | 0.820 |
| Pre-crash fields only | 0.665 | 0.259 | 0.249 | 0.508 | 0.688 |
| Always predict "no injury" | 0.500 | 0.154 | — | 0.000 | 0.846 |

### Severe injury model

| | AUC-ROC | PR-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| All features | 0.833 | 0.119 | 0.184 | 0.235 | 0.969 |
| Pre-crash fields only | 0.649 | 0.038 | 0.029 | 0.494 | 0.712 |
| Always predict "not severe" | 0.500 | 0.017 | — | 0.000 | 0.983 |

### Model complexity buys nothing

Injury target, same features, same split:

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic regression | 0.814 | 0.538 |
| Random forest | 0.814 | 0.533 |
| Gradient boosting | 0.812 | 0.534 |

Three models within 0.002 AUC of each other. The signal is in the features, not in the functional
form, so the deployed app uses the logistic model — it is the only one whose coefficients are
directly readable.

## What the leakage test shows

`PRIM_CONTRIBUTORY_CAUSE` and `FIRST_CRASH_TYPE` are recorded by the reporting officer *after* the
crash, on the same report that records the injury. Including them is legitimate for a descriptive
model and misleading for a predictive one.

Removing them costs 0.147 AUC on the injury target and 0.184 on the severe target, and PR-AUC on
severe injury collapses from 0.119 to 0.038 — barely above the 0.017 prevalence floor. Roughly
half of the apparent performance was reconstruction, not prediction.

**Read the top-line numbers as descriptive. The pre-crash rows are the honest predictive estimate.**

## Run locally

```
pip install -r requirements.txt
python3 chicago_crash_injury_model.py 
python3 app3.py                          # http://127.0.0.1:8052
```

## Known limitations

- **Post-hoc fields.** See above. The headline AUC is descriptive; the pre-crash figures are the
  number to quote if the question is prediction.
- **Reporting quality is uneven.** About half of crash reports are self-reported by drivers at the
  police district rather than recorded on scene. Weather, lighting and surface conditions are the
  officer's judgement from best available information and may disagree with actual conditions.
- **The split is random, not temporal.** Test performance is not an estimate of how the model would
  generalise to a future year, and `CRASH_MONTH` is used as a feature, so seasonal structure is
  partly learned from the same period it is evaluated on.
- **Severe-injury precision is low** (0.184 at the chosen threshold). At a 1.7% base rate, roughly
  four in five flagged crashes are false positives. The model ranks risk usefully; it does not
  classify reliably.
- **No causal claim.** Speed limit correlating with injury does not mean lowering the posted limit
  on a given street would reduce injuries — the roads with high limits differ from the roads with
  low limits in many other ways.
