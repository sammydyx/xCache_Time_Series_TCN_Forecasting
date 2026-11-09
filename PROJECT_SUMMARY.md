# SoCal XCache Daily Forecasting: Project Summary

## Research question

Can a Temporal Convolutional Network (TCN) forecast next-day SoCal XCache traffic while respecting the physical relationship

\[
\mathrm{access} = \mathrm{hit} + \mathrm{miss}
\]

for both request counts and transferred sizes? Beyond average-day RMSE, can historical traffic identify future high-traffic peaks and one-day spikes?

## Data and common protocol

- **Site:** SoCal XCache.
- **Daily period:** 2020-05-31 to 2024-03-31 (1,401 days).
- **Targets:** `access_count`, `access_size`, `hit_count`, `hit_size`, `miss_count`, and `miss_size`.
- **Forecast horizon:** one day ahead.
- **Units:** counts are requests; size variables are TB.
- **Splits:** chronological train/validation/test splits. Major change points motivate three regimes: early (through September 2021), transition (October 2021 through June 2022), and late/post-change-point (July 2022 onward).
- **Primary baseline:** persistence, i.e., tomorrow is predicted as today.

## Model family

### Hard Structured TCN

The model predicts log-access and a hit ratio for count and size:

\[
\hat{H}=\hat{A}\hat{r}, \qquad \hat{M}=\hat{A}(1-\hat{r}).
\]

It guarantees non-negative predictions and exact identity consistency:

\[
\hat{A}=\hat{H}+\hat{M}.
\]

The standard architecture used later in the project has five residual TCN blocks, 64 channels per block, kernel size 3, and dilations 1, 2, 4, 8, and 16. Its receptive field is 125 days.

### Other models studied

- **Direct TCN:** predicts all six variables independently; it can improve some RMSE values but does not guarantee the identity.
- **LSTM baseline:** 128 units, tanh activation, 0.04 dropout, 50 epochs; rerun on the SoCal data rather than compared directly with values reported in a previous paper.
- **Soft-constrained multi-task TCN:** predicts six variables directly and adds a weighted identity penalty.
- **Logit-ratio Structured TCN:** predicts log-access and logit-transformed hit ratio, while retaining the hard identity.

## Research progression and findings

### Week 1: Data exploration

- Explored daily traffic at Chicago, Boston, and SoCal.
- Identified that SoCal has distinct traffic regimes and that count and size variables behave differently.
- Established the one-day-ahead daily forecasting task.

### Week 2: Structured TCN baseline

- Built the first three-block, 28-day Structured TCN.
- Motivated the model using the access-hit-miss identity.
- Established RMSE in original units as the principal error measure.

### Week 3: Architecture, lookback, and baseline comparisons

- Clarified TCN channels, residual blocks, dilation, and receptive field.
- A three-block TCN has a 29-day receptive field; a five-block TCN has a 125-day receptive field.
- Scanned lookback windows from 7 to 120 days and compared direct and structured TCNs, a Week 2 reference, and an LSTM baseline on the full chronological split.
- Five residual blocks with 64 channels were selected as the standard architecture for later analysis.
- Key lesson: a longer supplied lookback is useful only if the architecture has a receptive field large enough to use it.

### Week 4: Change-point-aware post-change-point baseline

- Restricted training, validation, and test data to the late regime after the July 2022 change point.
- The 14-day, five-block, 64-channel hard Structured TCN beat persistence for five of six variables:

| Variable | Structured TCN RMSE | Persistence RMSE |
|---|---:|---:|
| access_count | 15,799.9 | 19,116.8 |
| access_size | 40.535 | 47.845 |
| hit_count | 14,847.8 | 17,914.6 |
| hit_size | 40.040 | 47.000 |
| miss_count | 1,261.5 | 1,482.9 |
| miss_size | 1.722 | 1.156 |

- Calendar features (day of week, annual seasonality, weekend) were also evaluated. They did not become the preferred model because their improvements were not stable across all variables.

### Week 5: The cost and value of the hard identity constraint

- Verified that the hard Structured TCN has essentially zero identity error by construction.
- Derived the miss-error decomposition:

\[
e_M=(1-r)e_A-\hat{A}e_r.
\]

- Found that hit-ratio error often dominates miss error, especially for size when the hit ratio is near one.
- The hard raw-ratio Structured TCN remained strongest for the primary access/hit variables in the post-change-point experiment.
- Soft-constrained and logit-ratio variants improved `miss_size` but degraded one or more access/hit metrics. For example, logit-ratio Structured TCN reduced `miss_size` RMSE to 0.950 TB, below persistence (1.156 TB), while the raw-ratio hard model remained better for access and hit forecasting.
- Main conclusion: the structural identity supplies useful inductive bias and exact feasibility, but it creates a real accuracy tradeoff among access, hit, and miss targets.

### Week 6: Peak, spike, and traffic-regime analysis

- Defined a peak as a target above the training 90th percentile and a spike as a one-day increase above the training 90th percentile.
- Evaluated normal days, peaks, and spikes across early, transition, and late regimes with 14-, 56-, and 112-day lookbacks.
- Peak and spike RMSE were much higher than normal-day RMSE, with systematic negative bias on high-traffic days.
- Daily-history models frequently smoothed extreme events rather than matching their magnitude.
- Regime transfer was difficult: performance degraded sharply when train and test periods differed in traffic behavior.

### Week 7: Causal trend features and robustness

- Added 12 causal `hit_size` trend features: moving-average levels, relative moving-average gaps, EMA gaps, and 3-/7-day slopes.
- Trend features did not provide stable held-out gains over the 28-day daily baseline.
- Re-evaluated event detection for both `hit_count` and `hit_size`.
- `hit_count` showed more detectable event signal than `hit_size` in the late-regime held-out period. For the 28-day base model, hit-count peak recall was 64.3% and spike recall was 31.3% in the initial single-run comparison, whereas hit-size spike recall was zero.
- Five-seed regime experiments showed substantial dependence on traffic regime and initialization. Therefore, single-seed event results should be treated as exploratory.

### Week 8: Longer lookbacks and hourly intraday summaries

- Compared daily-only 14-, 28-, 56-, and 112-day Structured TCNs for `hit_count` in the late regime.
- There is no single universally best lookback:

| Objective | Best single-run lookback | Result |
|---|---:|---:|
| Validation hit-count RMSE | 28 days | 7,955.9 |
| Test hit-count RMSE | 56 days | 14,324.4 |
| Test peak recall | 112 days | 42.9% |
| Test spike recall | 112 days | 12.5% |
| Test spike Top-K recall | 112 days | 31.3% |

- Constructed causal hourly intraday summaries from every completed historical day. The expanded model adds 24 features: hourly high, low, mean, and range for each of the six variables.
- At a 28-day lookback, all-variable intraday summaries improved validation hit-count RMSE and event detection, but worsened held-out overall hit-count RMSE:

| Metric | Daily-only | Daily + intraday summaries |
|---|---:|---:|
| Validation hit-count RMSE | 7,955.9 | 7,425.2 |
| Test hit-count RMSE | 14,874.6 | 15,555.8 |
| Peak recall | 21.4% | 42.9% |
| Spike recall | 0.0% | 25.0% |

- The results suggest an average-error versus event-sensitivity tradeoff. They are not yet robust enough for a final claim because they are based on one seed and one held-out period.

## Current best results: precise interpretation

There is **not one model that is best for every criterion**. The paper should report a model family and task-dependent tradeoffs rather than claim a universal winner.

1. **Best physically consistent primary baseline:** five-block hard Structured TCN, raw hit ratio, 64 channels per block. It guarantees non-negative outputs and exact `access = hit + miss` consistency.
2. **Best post-change-point primary-variable result:** the hard Structured TCN beat persistence for access_count, access_size, hit_count, hit_size, and miss_count; it did not beat persistence on miss_size.
3. **Best miss-size result among tested variants:** logit-ratio Structured TCN, with the tradeoff of worse primary access/hit results.
4. **Best single-run overall hit-count RMSE in the latest long-lookback scan:** 56-day daily-only model.
5. **Best single-run event recall in the latest long-lookback scan:** 112-day daily-only model.

## What can be claimed now

- Incorporating the access-hit-miss identity produces feasible predictions with zero identity error.
- The hard Structured TCN is competitive with, and in the late regime improves on, persistence for the primary access/hit traffic measures.
- Constraint design changes the distribution of errors: improving consistency or miss-size accuracy can reduce access/hit accuracy.
- Peak and spike forecasting is substantially harder than normal-day forecasting from daily history alone.
- Lookback length and feature engineering are task-dependent; they do not have a single monotonic benefit.
- Traffic regime shift materially affects forecast performance.

## Claims that require further validation

- That 112 days is truly best for peak/spike forecasting.
- That hourly intraday summaries reliably improve event detection.
- That any feature set is superior across random seeds and regimes.

## Recommended remaining experiments before final paper results

1. Run 28-, 56-, and 112-day daily-only candidates over multiple seeds; report median and IQR for RMSE and event metrics.
2. Choose a validation-only selection rule for the event-oriented task before inspecting the final test set.
3. Replace redundant hourly means with compact relative shape features, for example high/mean, low/mean, and range/mean; daily totals already contain the level information.
4. Compare daily-only and compact intraday models across multiple seeds for the selected long-lookback candidate.
5. Use a final held-out test period only once after the selection rule and final configuration are fixed.

## Paper structure

1. **Introduction:** operational importance of XCache traffic forecasting; structural access-hit-miss relationship; difficulty of regime shifts and extreme traffic days.
2. **Data and task:** SoCal data, six variables, daily one-day-ahead task, units, chronological splits, change points.
3. **Methods:** persistence and LSTM baselines; direct versus hard Structured TCN; receptive field; soft/logit variants; evaluation metrics.
4. **Main results:** post-change-point hard Structured TCN versus persistence; include all six RMSE values and identity error.
5. **Analysis:** constraint tradeoff, regime dependence, lookback effects, and peak/spike error.
6. **Feature experiments:** trend and hourly intraday features as exploratory analyses, clearly labeled as such unless multi-seed validation is completed.
7. **Limitations and future work:** lack of workflow/job/cache-health covariates; daily data alone may not contain precursor information for rare spikes.

## Poster structure

1. **Problem and data:** one-day-ahead SoCal XCache forecasting; six outputs and the physical identity.
2. **Method diagram:** five-block Structured TCN with log-access and hit-ratio outputs, followed by deterministic hit/miss reconstruction.
3. **Main result table:** post-change-point Structured TCN versus persistence for all six variables.
4. **Key analysis figure:** normal/peak/spike error or the long-lookback tradeoff plot.
5. **Takeaway box:** exact structural consistency; improved primary traffic forecasts; event forecasting remains difficult and regime-dependent.
6. **Future work:** robust multi-seed event evaluation and compact intraday shape features.
