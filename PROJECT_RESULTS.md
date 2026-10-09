# Project Results: Weather Foundation Model → Electricity Demand and Price

Running record of what has been tested, exactly how, and what came out.
Setup and cluster commands live in `PROJECT_NOTES.md`.

Last updated: 2026-10-09

## Status

| Model | Status |
|---|---|
| Calendar only | Done (provisional) |
| Calendar + simple weather | Done (provisional) |
| Calendar + Prithvi embedding | **Not run.** First embedding table was all NaN; cluster rerun in progress. |
| Calendar + weather + Prithvi | **Not run**, same reason. |

"Provisional" because the weather features will change slightly once the
MERRA-2 daily files are rebuilt (see "Known issues").

Nothing here says anything yet about whether Prithvi helps.

Everything down to the divider is the **demand** models. The **price** models, added
2026-10-09, are in their own section at the end.

## The question

Does a frozen Prithvi WxC representation of the regional atmosphere add
information for predicting electricity demand, beyond calendar features and
a few conventional weather variables?

## What is being predicted

- **Target:** hourly real-time electricity **demand (load), in MW**. Not
  price. Not a forecast product: this is ISO-NE's reported actual load.
- **Source:** ISO-NE web services, `realtimehourlydemand`, field `Load`.
- **Zones:** all eight ISO-NE load zones: CT, ME, NH, RI, VT, and the three
  Massachusetts zones NEMA (Boston), SEMA, WCMA.
- **Period:** calendar year 2024 in Eastern local time, 8,784 hours per zone.
- **Rows:** 70,272 zone-hours, of which 70,224 are used. The 48 dropped are
  the last 6 local hours of Dec 31 in each zone, which fall in 2025 UTC and
  have no matching weather.
- The target is raw MW. No log transform, no per-zone normalisation.

Mean demand by zone (MW): CT 3,041 · NEMA 2,674 · WCMA 1,790 · SEMA 1,585 ·
NH 1,295 · ME 1,252 · RI 870 · VT 544.

## What the models are given

Every model is given the calendar features. The feature sets differ only in
what weather information is added.

**Calendar (7 features), from Eastern local time:**
`hour`, `day_of_week`, `day_of_year`, `month`, `is_weekend`, `is_holiday`
(US federal holidays, 11 days in 2024), `zone` (categorical).

**Simple weather (6 features), each row's own zone:**
`T2M` (2 m air temperature, K), `QV2M` (2 m specific humidity),
`wind_speed`, `U10M`, `V10M` (10 m wind, m/s), `SWGNT` (surface net
shortwave, W/m²).

- Source: MERRA-2, the value at the single grid cell nearest a hand-picked
  point for each zone. The grid is about 50 km, so the three Massachusetts
  zones are only a cell or two apart.
- Weather is 3-hourly (00, 03, ... 21 UTC). Each hourly demand row is given
  the nearest weather time, at most 90 minutes away. So three consecutive
  hours share one weather value.

**Prithvi (2,560 features, not yet run):**
the global-pooled encoder embedding, `global_emb_0` ... `global_emb_2559`.
One vector per 3-hourly weather time, the same for all eight zones, joined
to demand the same way as the simple weather.

**Not given to any model:** past demand (no lags, no yesterday's or last
week's load), demand forecasts, prices, or anything from after the hour
being predicted. This is deliberate: the comparison is between feature
sets, not an attempt at an operational load forecast. Operational forecasts
that use lagged load reach errors of a few percent.

## Model setup

- **Algorithm:** CatBoost regression (`CatBoostRegressor`, version 1.2.10).
- **One pooled model per feature set**, trained on all eight zones together
  with `zone` as a categorical feature. No per-zone models yet.
- **Settings:** RMSE loss, depth 6, learning rate 0.05, up to 3,000 trees,
  seed 7. All other settings are CatBoost defaults.
- **No hyperparameter tuning.** The same settings are used for every
  feature set and split.
- **Early stopping:** training stops when validation error has not improved
  for 100 trees, and the best tree count is kept. The validation block is
  separate from the test block; the test block is never seen during fitting.
- One run per model. No repeated seeds, so there are no error bars yet.

## Train / validation / test splits

Splits are by Eastern local date. Rows are never split at random, because
demand in adjacent hours is strongly correlated.

| Split | Train | Validation | Test |
|---|---|---|---|
| **Chronological** | Jan–Aug (46,840 rows) | Sep (5,760) | Oct–Dec (17,624) |
| **Blocked** | days 1–18 of each month (41,472) | days 19–21 (6,912) | day 22 onward (21,840) |

- **Chronological** is the stricter test: a later, contiguous period. Its
  weakness with one year of data is that the test months come after
  everything in training, so `day_of_year` and `month` carry no usable
  information there, and the late-year cold season has only January–March
  to learn from.
- **Blocked** puts every season in both train and test. Its weakness is
  that test days sit next to training days, so slowly varying conditions
  leak across the boundary and scores are optimistic.

The truth is probably between the two. More years of data would remove the
need to choose.

## How results are scored

- **MAPE:** mean of |predicted − actual| / actual, in percent, per zone.
- **R²:** per zone.
- **Headline numbers are the mean over the eight zones.** Pooled R² across
  zones is not used as a headline because zone size alone explains most of
  the pooled variance (calendar-only scores 0.85 pooled on a split where it
  is worse than a constant within most zones).
- **Conditions**, each defined within a zone on the test set: high demand
  (top 10% of hours), hot (top 10% T2M), cold (bottom 10% T2M). Bias is the
  mean of (predicted − actual) / actual; negative means under-prediction.

## Results so far (provisional)

### Headline

| Split | Feature set | Mean zone MAPE | Mean zone R² | Trees |
|---|---|---|---|---|
| Chronological | Calendar | 15.2% | −0.12 | 295 |
| Chronological | Calendar + weather | 9.0% | 0.53 | 348 |
| Blocked | Calendar | 11.0% | 0.46 | 2,997 |
| Blocked | Calendar + weather | 5.9% | 0.84 | 388 |

### MAPE by zone (%)

| Zone | Chrono: calendar | Chrono: + weather | Blocked: calendar | Blocked: + weather |
|---|---|---|---|---|
| CT | 13.3 | 9.5 | 9.5 | 5.5 |
| ME | 12.3 | 9.1 | 8.1 | 4.8 |
| NEMA | 20.5 | 7.1 | 12.3 | 4.7 |
| NH | 11.5 | 7.5 | 8.8 | 4.8 |
| RI | 14.8 | 8.6 | 11.1 | 5.5 |
| SEMA | 17.6 | 9.5 | 10.0 | 5.3 |
| VT | 19.2 | 12.8 | 19.2 | 12.0 |
| WCMA | 12.1 | 8.2 | 9.0 | 5.1 |

### By condition (MAPE %, with bias % in brackets)

| Condition | Chrono: calendar | Chrono: + weather | Blocked: calendar | Blocked: + weather |
|---|---|---|---|---|
| High demand | 13.2 (−9.8) | 16.1 (−16.1) | 12.5 (+0.7) | 6.5 (−1.0) |
| Hot | 25.3 (+24.8) | 9.3 (+7.8) | 14.6 (+6.4) | 7.4 (+3.7) |
| Cold | 16.4 (−15.3) | 15.6 (−15.6) | 10.4 (−5.4) | 6.1 (−1.6) |

### What these show

- **Simple weather helps substantially**, in every zone and on both splits.
  It roughly halves the error on the blocked split.
- **Calendar alone fails on the chronological split** (negative mean R²):
  it cannot know that October–December differs from late August.
- **Cold and peak hours are under-predicted by about 16% on the
  chronological split even with weather.** The test period's cold hours are
  colder-season conditions the model saw little of. Weather does not fix
  this there; on the blocked split the same conditions are predicted well.
- **Vermont is the outlier** (about 12% with weather, against 5–9%
  elsewhere). Not investigated.
- `zone` dominates feature importance (about 60%) because it sets the
  overall level. Among the rest, `hour`, `T2M`, and `QV2M` lead.

## Assumptions and limitations

- One year of data. No result here has been checked on a second year.
- Single run per model, untuned. Small differences between models should
  not be read as real until there are repeated seeds or more data.
- Zone weather is one grid cell at a hand-picked point, not a
  population- or load-weighted average.
- 3-hourly weather against hourly demand.
- The Prithvi embedding is one vector for the whole region, identical
  across zones. Any zone-specific signal has to come from its interaction
  with the `zone` feature.
- With 2,560 embedding columns and about 2,900 distinct weather times, the
  Prithvi models will have far more features per independent weather state
  than the baselines. Overfitting is a real risk and the validation block
  is the only guard so far. A reduced embedding (for example PCA) is the
  obvious follow-up comparison.

## Known issues

- **Embeddings all NaN (2026-10-07).** Cause, inferred from the repo's own
  input-preparation code: `GWETROOT` and `LAI` are undefined over ocean and
  were not filled. Fixed in `src/merra2/build_daily_prithvi_files.py`; not
  yet confirmed on real data.
- **Weather features are provisional.** The same fix switches the
  single-level variables (including T2M) from hourly averages to
  instantaneous values, so the baseline numbers above will move slightly
  when rerun.

## How to reproduce

```bash
.venv/bin/python local/prepare_modeling_table.py \
  --compact-parquet data/compact/modeling_input_2024.parquet \
  --out-parquet data/modeling_table_2024.parquet
.venv/bin/python local/train_catboost_comparisons.py \
  --modeling-table data/modeling_table_2024.parquet \
  --out-dir results/phase3_baselines_2024
```

Outputs (not in git): `comparison_summary.csv`, `comparison_by_zone.csv`,
`comparison_by_condition.csv`, `feature_importance_*.csv`,
`test_predictions.parquet`.

## Still to do

- Rerun all four feature sets once finite embeddings are back.
- Observed-vs-predicted plots.
- Repeated seeds for error bars.
- Reduced-dimension Prithvi embedding; per-region embeddings.
- Demand models: per-zone models; Vermont's midday over-prediction.

---

# Price models (first pass, 2026-10-09)

Separate from everything above. No Prithvi features yet.

## What is being predicted

- **Target:** the hourly price at the ISO-NE internal **hub**, total LMP in
  **$/MWh**. Two versions, modelled separately:
  - **Day-ahead** (`hourlylmp/da/final`): set the previous day.
  - **Real-time** (`hourlylmp/rt/final`): settled from actual conditions.
- **Why the hub and not the eight zones:** zonal prices are almost one
  series. In 2024 every zone's day-ahead price correlates 0.995 or higher
  with the hub and differs from it by $0.64/MWh on average. Modelling eight
  zones would be eight copies of the same target.
- **Period and rows:** 2024 in Eastern local time, one row per hour, 8,778
  hours used (6 late-Dec-31 hours have no weather).

2024 hub prices, for scale:

| | Mean | Median | 1st–99th percentile | Max | Negative hours |
|---|---|---|---|---|---|
| Day-ahead | $41.47 | $32.42 | $16–$170 | $327 | 0 |
| Real-time | $39.50 | $30.05 | $10–$168 | $2,113 | 29 |

Prices are strongly seasonal and spiky: the day-ahead monthly mean was $70
in January and $88 in December, against $24–$46 in every other month.
Day-ahead and real-time correlate only 0.62 hour by hour.

## What the models are given

| Feature set | Features |
|---|---|
| Calendar | hour, day_of_week, day_of_year, month, is_weekend, is_holiday |
| Calendar + weather | + regional T2M, QV2M, wind speed, U10M, V10M, SWGNT |
| Calendar + weather + demand | + actual system demand (eight zones summed) for the same hour |

- **Regional weather** is the eight zones' weather averaged with fixed
  weights equal to each zone's share of annual demand.
- **Demand is actual demand, not predicted.** That feature set is an upper
  bound on what a demand prediction could add. It is not a usable forecast
  input.
- **Everything is from the same hour as the price.** These are explanatory
  models, not forecasts: the day-ahead price is really set a day earlier
  from forecasts.
- **Not given:** natural gas or other fuel prices, past prices, generator
  outages, imports, renewable output.

## Model setup, splits, scoring

- CatBoost with the same settings as the demand models (RMSE loss, depth 6,
  learning rate 0.05, up to 3,000 trees, early stopping on the validation
  block, seed 7, no tuning, one run).
- The same two splits: chronological (train Jan–Aug, validate Sep, test
  Oct–Dec; 5,855 / 720 / 2,203 hours) and blocked (days 1–18 / 19–21 / 22
  onward of each month; 5,184 / 864 / 2,730 hours).
- **Scored in $/MWh**: mean absolute error (MAE), median absolute error,
  bias (mean of predicted − actual), and R². No percentage error, because
  prices approach zero and real-time prices go negative.
- **Reference:** always predicting the training-period mean price.

## Results

### Blocked split

| Feature set | Day-ahead MAE | median | R² | Real-time MAE | median | R² |
|---|---|---|---|---|---|---|
| Training mean price | $17.35 | $12.60 | 0.00 | $17.64 | $13.99 | −0.01 |
| Calendar | $19.37 | $6.14 | −0.52 | $16.91 | $9.34 | 0.06 |
| Calendar + weather | $12.11 | $4.54 | 0.39 | $14.21 | $8.78 | 0.33 |
| Calendar + weather + demand | $9.86 | $3.57 | 0.56 | $11.70 | $8.11 | 0.53 |

### Chronological split

| Feature set | Day-ahead MAE | bias | R² | Real-time MAE | bias | R² |
|---|---|---|---|---|---|---|
| Training mean price | $23.99 | −$16.83 | −0.23 | $25.44 | −$17.97 | −0.23 |
| Calendar | $24.71 | −$22.98 | −0.37 | $24.79 | −$19.49 | −0.24 |
| Calendar + weather | $24.85 | −$23.98 | −0.27 | $25.43 | −$21.53 | −0.29 |
| Calendar + weather + demand | $24.21 | −$21.70 | −0.23 | $25.14 | −$21.28 | −0.24 |

### Day-ahead, blocked split, by condition (MAE, with bias in brackets)

| Condition | Calendar | + weather | + weather + demand |
|---|---|---|---|
| Ordinary price (middle 80%) | $16.0 (+12.0) | $9.4 (+5.6) | $7.1 (+3.8) |
| High price (top 10%) | $60.8 (−29.6) | $42.3 (−18.6) | $38.6 (−19.3) |
| Hot (top 10% T2M) | $27.6 (+22.9) | $9.9 (+5.7) | $6.4 (+2.8) |
| Cold (bottom 10% T2M) | $53.4 (−22.3) | $40.7 (−5.9) | $35.6 (−9.8) |

### What these show

- **On the chronological split nothing works.** No feature set beats
  predicting the training mean. The test period contains December, when the
  day-ahead price averaged $88; the best model predicted $36 for that
  month. October and November are under-predicted by $5–$8.
- **On the blocked split, weather and demand both help.** For day-ahead,
  weather takes R² from below zero to 0.39 and actual demand to 0.56.
- **The typical hour is predicted far better than the average suggests.**
  Day-ahead median error with weather and demand is $3.57 against a mean of
  $9.86. Unlike the demand models, the mean here is driven by a minority of
  expensive hours: the top 10% of prices are under-predicted by about $19
  with a $39 mean error.
- **Cold hours are the hard case**, hot hours are not. Winter price spikes
  in New England follow natural gas prices, which no model here sees. That
  is the most likely reason for both the December failure and the cold-hour
  errors, but it has not been tested with fuel-price data.
- **Real-time is harder than day-ahead** in the typical hour (median error
  $8 against $3.57), as expected for the noisier series.

## Price-specific limitations

- One year: a single January and a single December carry all the
  information about winter price spikes, and the chronological split puts
  December entirely in the test set.
- 8,778 hourly rows, a tenth of the demand table.
- RMSE loss on a spiky target. A log or robust loss was not tried.
- Actual demand as a feature, same-hour inputs, no fuel prices (see above).

## How to reproduce (price)

```bash
for s in dayahead realtime; do
  .venv/bin/python local/prepare_price_table.py \
    --lmp-parquet data/compact/isone_${s}_hourly_lmp_2024.parquet \
    --modeling-table data/modeling_table_2024.parquet \
    --out-parquet data/price_table_${s}_2024.parquet
  .venv/bin/python local/train_price_catboost.py \
    --price-table data/price_table_${s}_2024.parquet \
    --out-dir results/price_${s}_2024
done
```

## Price: still to do

- Prithvi feature sets, once finite embeddings are back.
- Predicted (not actual) demand as a feature, with out-of-sample
  predictions for the training rows.
- A natural gas price series, to test the fuel-price explanation.
- Day-ahead-appropriate inputs (previous-day information only).
