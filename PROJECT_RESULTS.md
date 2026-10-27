# Project Results: Weather Foundation Model → Electricity Demand and Price

Running record of what has been tested, exactly how, and what came out.
Setup and cluster commands live in `PROJECT_NOTES.md`.

Last updated: 2026-10-09

## Status (2026-10-13, late)

**Every Prithvi number in this file so far came from the model running at
reduced accuracy, on a regional crop it does not work well with.** See
"Regional-crop problem" below. Extraction has been moved to the global grid
and confirmed on a two-day test; full years are being rerun. The calendar
and simple-weather results are unaffected.

| Item | Status |
|---|---|
| Calendar, calendar + weather | Valid. Three years (2022-2024). |
| Pooled Prithvi embedding (all earlier results) | From the degraded regional-crop model. Indicative only. |
| Local patch tokens (earlier result) | Invalid: degraded model, and tokens in the wrong quadrant. |
| Global-grid extraction | Working on a two-day test. 2020 in progress; 2021-2024 need global downloads. |
| Decoder output, earlier encoder layers, raw input cube as baseline | Saved by the new extraction; analysis waits for the first global year. |

## Regional-crop problem (found 2026-10-13)

All extraction up to this date gave the model a 60 x 64 pixel crop
(30-60N, 90-50W; 2 x 2 mask units) instead of the 360 x 576 global grid it
was trained on. The code accepts any grid whose size is a multiple of the
mask unit, and this was taken as evidence that cropping was valid. It was
never tested until now.

**Test** (`src/prithvi/sample_validation.py`): the model is run on NASA's
own global sample data (2020) and its output scored against the target,
relative to the climatology it is given: skill = 1 - MSE(output - target) /
MSE(climatology - target). At lead time 0 the target equals the input, so a
working model should score near 1. The same four samples are cropped to
areas of increasing size around New England; every row is scored in the
same small New England box (40-47.5N, 75-66.9W).

| Area given to the model | Skill, lead 0 h | Skill, lead 6 h | T2M error, lead 0 (sd) |
|---|---|---|---|
| 2 x 2 mask units (the crop used so far) | 0.82 | 0.44 | 0.086 |
| 4 x 4 (15-75N, 110-30W) | 0.94 | 0.54 | 0.044 |
| 4 x 6 (15-75N, 130-10W) | 0.94 | 0.60 | 0.037 |
| 6 x 8 (0-90N, 150W-10E) | 0.96 | 0.60 | 0.028 |
| 12 x 18 (global) | 0.98 | 0.95 | 0.013 |

Skill is the median over the 20 surface variables; vertical variables show
the same pattern. Four samples, one date range (2020-01-01 to 03).

- **The model works as released on the global grid** (0.98 at lead 0, 0.95
  at 6 h), so the weights, loading and data handling are right.
- **It needs global context.** Skill rises with the area but only the full
  grid gets close to 1; a quarter of the globe still gives 0.60 at 6 h.
- **Block order does not affect skill** (shuffled, as in the repo's default
  forward pass, and map order score the same on the global grid). It only
  affects where saved tokens sit.
- **The shifted-window attention matters**: switching it off on the global
  grid drops 6 h skill from 0.96 to 0.78.

**Fix:** extraction now runs on the global grid and saves the same New
England patch tokens. On a two-day 2020 test the decoder's T2M error was
0.017 sd against 0.147 sd for climatology (it was 0.055 against 0.059 on
the regional crop). 2020 uses the global daily files shipped with the model
weights; other years need global MERRA-2 downloads (about 150 GB per year).

## Block-order bug (found 2026-10-13)

- **What:** Prithvi groups its tokens into large blocks ("mask units"; our
  crop has 2 x 2). With masking switched off, the model's mask generator
  still shuffles those blocks into a random order for every sample before
  the encoder. The extraction used the model's standard forward pass, with
  settings copied from the repo's inference notebook, and inherited this.
- **Effect on the encoder:** none measurable. Skill on the global grid is
  the same with shuffled and map-ordered blocks (see "Regional-crop
  problem"). An earlier version of this note said the attention mixed the
  wrong neighbours; the test does not support that.
- **Effect on saved outputs:** the crop-pooled embedding is an average, so
  the shuffle does not change what was averaged, but the values were
  produced by the mis-ordered encoder. The four quadrant embeddings and all
  patch tokens were assigned to the wrong place three times in four.
- **Evidence** (2024 tokens): anomaly similarity between adjacent patches
  is 0.72 inside a block and 0.09 across a block boundary (0.04 for
  far-apart patches); a patch compared with itself 3 hours later matches
  closely about a quarter of the time and not at all otherwise.
- **Fix:** `extract_embeddings.py` now forces map order, as the repo's own
  `masked_inference()` does. Not yet rerun.
- **Consequence:** the "Local-token results" section below should be
  disregarded, including its conclusion that local temperature is hard to
  read from a patch token. The pooled results in "First Prithvi results"
  are kept as indicative.

## Nowcast or forecast? Information-lag experiment (2026-10-13)

Every result elsewhere in this file is a **nowcast with perfect
information**: weather features and the Prithvi input are reanalysis for
the same hour as the demand being predicted. That favours local weather,
since demand at an hour is mostly a function of temperature at that hour.

This experiment gives every model only a description of the atmosphere
L hours before the demand hour (calendar features still describe the
demand hour). Train 2022 + 2023, test 2024, 3,000 trees, one seed. Pooled
Prithvi embedding (50 components), from the flawed extraction.
Script: `local/experiment_information_lag.py`.

Mean zone MAPE:

| Inputs | Same hour (L = 0) | 24 h earlier | 48 h earlier |
|---|---|---|---|
| Calendar | 11.5% | 11.5% | 11.5% |
| Calendar + weather | 4.85% | 7.12% | 8.66% |
| Calendar + pooled Prithvi | 8.80% | 8.74% | 9.08% |
| Calendar + weather + pooled Prithvi | 5.08% | 6.78% | 7.78% |

- **Local weather loses value quickly as it ages** (4.9% to 7.1% to 8.7%).
- **The pooled embedding barely changes** (8.8%, 8.7%, 9.1%): it describes
  the large-scale state, which persists.
- **The sign of Prithvi's contribution flips.** Added to weather it hurts
  at the same hour (+0.2 points) and helps with older information (−0.3
  points at 24 h, −0.9 at 48 h). At 48 h it is nearly as good alone as
  weather alone.
- **This is not a forecast comparison.** "Weather 24 h earlier" is a
  persistence baseline. A real day-ahead test would give the weather arm
  an actual 24-hour forecast, which is far better than yesterday's
  weather, and would run Prithvi with a 24-hour lead time. Neither is done.
- With three years, the same-hour weather baseline improves from 5.3%
  (train 2023 only) to 4.85%.

## First Prithvi results (2026-10-12)

### What changed from the earlier runs

- **Two years.** 2023 and 2024 are complete (2022 is still downloading).
- **New headline split, "year-ahead":** train on all of 2023, test on all of
  2024 (70,032 training and 70,224 test zone-hours). No test day is near a
  training day, and every season is in both.
- **Fixed 3,000 trees, no early stopping** (see "Does training longer
  help?"). September and days 19-21 are still left out of the two
  2024-only splits so those stay comparable with earlier runs.
- **Rebuilt weather features:** instantaneous values from M2I1NXASM.
- **Prithvi embedding:** the encoder output averaged over every token in the
  model's input crop (30-60N, 90-50W, an area running from Georgia to Hudson
  Bay), one 2,560-vector per 3-hourly weather time, the same for all zones.
  Standardised and reduced to 50 principal components (95% of its
  variance), fitted on training rows only.

### Results: mean absolute percentage error, mean over the eight zones

| Feature set | Year-ahead (train 2023, test 2024) | Chronological (2024) | Blocked (2024) |
|---|---|---|---|
| Calendar | 11.9% | 15.5% | 11.0% |
| Calendar + weather | **5.3%** | **8.9%** | **5.3%** |
| Calendar + Prithvi (pooled) | 9.9% | 17.0% | 9.6% |
| Calendar + weather + Prithvi (pooled) | 5.7% | 9.2% | 5.8% |

Year-ahead mean zone R²: 0.43, 0.88, 0.61, 0.86 in the same order.

Year-ahead, by zone (MAPE %):

| Zone | Calendar | + weather | + Prithvi | + weather + Prithvi |
|---|---|---|---|---|
| CT | 11.3 | 4.8 | 10.3 | 5.3 |
| ME | 8.4 | 5.2 | 7.4 | 5.4 |
| NEMA | 10.4 | 4.0 | 9.4 | 4.4 |
| NH | 7.9 | 4.2 | 7.3 | 4.7 |
| RI | 14.1 | 4.9 | 9.9 | 5.3 |
| SEMA | 9.9 | 4.8 | 9.0 | 5.5 |
| VT | 24.3 | 9.8 | 18.2 | 9.9 |
| WCMA | 9.1 | 4.4 | 7.7 | 4.9 |

### What these show

- **The pooled Prithvi embedding carries real signal**: it cuts error from
  11.9% to 9.9% over calendar alone on the year-ahead split.
- **It is much weaker than six simple local weather numbers** (5.3%), and
  adding it to them makes the model slightly worse (5.7%), in every zone
  and on every split. On the chronological split it is worse than calendar
  alone.
- **This is not sensitive to how the embedding is reduced.** With 10, 20,
  50 or 200 components, or with the four quadrant embeddings instead of the
  single pooled one, year-ahead error stays between 10.2% and 10.8%
  (1,000-tree runs).

### Why, most likely: the embedding is pooled over too large an area

- A linear readout from the pooled embedding recovers the **crop-wide**
  mean temperature almost exactly (held-out R² 0.99), humidity (0.997),
  sunlight (0.98), time of day and season. The model input preparation is
  now sound and the encoder is clearly representing the weather.
- The same readout recovers an individual **zone's** temperature only to
  within about 2.3-3.2 K (RMSE, 2024, trained on 2023), and local wind
  hardly at all. The weather baseline has each zone's own temperature
  exactly.
- So this comparison is between local weather and a continental average,
  which is a test of the pooling choice, not of what Prithvi knows.

This explanation fits the evidence but has not been tested. The test is to
use the encoder's un-pooled patch tokens (each covers about 1 x 1.25
degrees) at each zone's location. `extract_embeddings.py` now saves these
for a New England box; it needs one more GPU pass per year.

## Local-token results (2026-10-13) -- INVALID, see "Block-order bug"

Tests the explanation above. The encoder's un-pooled patch tokens (one per
1 x 1.25 degree patch) were saved for an 8 x 7 patch box over New England,
for 2023 and 2024. Each zone is given the token(s) at its own location,
standardised and reduced to 50 principal components fitted on 2023. One
pooled CatBoost model, 3,000 trees, train 2023, test 2024.
Script: `local/explore_local_tokens.py`.

Mean zone MAPE (single seed unless noted):

| Prithvi input | Calendar + Prithvi | Calendar + weather + Prithvi |
|---|---|---|
| (none) | 11.9% | 5.26% (3 seeds: 5.25-5.27) |
| Pooled over the whole crop (earlier result) | 9.9% | 5.7% |
| Mean of all 56 New England patches | 10.0% | 5.16% |
| Nearest single patch | 9.1% | 5.18% |
| Four nearest patches, interpolated | 8.9% | 5.12% |
| Mean of the 3 x 3 patches around the zone | 8.8% (3 seeds: 8.82-8.84) | 5.11% (3 seeds: 5.09-5.13) |

Variations on the nearest patch: 20 components 9.3%, 200 components 9.1%.
All 2,560 raw dimensions with no PCA (3 x 3, 1,000 trees): 8.95% alone and
5.66% with weather.

### What these show

- **Local tokens are better than the pooled embedding**, but not by much:
  Prithvi alone improves from 9.9% to about 8.8%. Simple local weather is
  at 5.3%. The pooling was part of the problem, not most of it.
- **Local tokens add a small, repeatable gain on top of weather:** 5.26% to
  5.11%, with the three seeds of each not overlapping. It is 0.15 points.
- **More components or the raw 2,560 dimensions do not help.**

### Why local tokens fall short: local temperature is not easy to read from them

How well a straight-line fit recovers a zone's own 2 m temperature (fit on
2023, error on 2024):

| Source | Typical error (RMSE) |
|---|---|
| Crop-pooled embedding, 50 components | 2.8 K |
| Mean of all 56 New England patches, 50 components | 3.6 K |
| 3 x 3 patches around the zone, 50 components | 4.9 K |
| Nearest patch, 50 components | 6.2 K |
| Nearest patch, 1,000 components (Boston zone) | 3.4 K |
| Nearest patch, all 2,560 dimensions, ridge (Boston zone) | 3.7 K |

- A single patch token gives a zone's temperature **less** accurately than
  the big average does. Averaging many tokens sharpens the large-scale
  temperature; one token on its own is dominated by other content.
- There is no sign that a zone's temperature is held at that zone's
  patch: for all eight zones the best single patch for reading temperature
  was the same offshore one.
- The token layout itself was checked and is right: time-averaged tokens
  form smooth fields on the map with breaks exactly at the model's
  mask-unit boundaries, and neighbouring patches' anomalies are far more
  alike (0.60) than distant ones (0.05).

So the encoder output, whether pooled or local, does not hand a downstream
model the near-surface temperature precisely, and near-surface temperature
is most of what demand needs. This is a statement about these frozen
encoder tokens read with PCA or trees. It does not rule out other ways of
using the model (a different layer, the decoder's reconstructed fields, or
fine-tuning).

### Not yet done for these models

- No repeated seeds. Differences of a few tenths of a point are not
  reliable; the gap between weather (5.3%) and pooled Prithvi (9.9%) is far
  larger than that.
- The fixed 3,000-tree setting was checked for overfitting only on the
  calendar and weather models, not on the Prithvi ones.
- Price models have not been rerun with two years or with Prithvi.

Outputs: `results/phase3_2023_2024/`. Reproduce with:

```bash
for y in 2023 2024; do .venv/bin/python local/prepare_modeling_table.py \
  --compact-parquet data/compact/modeling_input_${y}.parquet --out-parquet data/modeling_table_${y}.parquet; done
.venv/bin/python local/train_catboost_comparisons.py \
  --modeling-table data/modeling_table_2023.parquet data/modeling_table_2024.parquet \
  --out-dir results/phase3_2023_2024
```

---

# Original 2024-only baselines (2026-10-07)

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
- **Second embedding table unusable (2026-10-10).** Values were finite but
  in the hundreds of millions, and five directions carried essentially all
  the variation across the year. Cause: the daily files stored the vertical
  levels in MERRA-2's native ascending order, while the loader assumes
  descending and flips them, so every vertical profile reached the model
  upside down. Fixed in `build_daily_prithvi_files.py` and checked against
  the loader's own level logic on synthetic files; extraction now also
  prints how many standard deviations each input sits from the model's
  training mean and stops if any is far out. Not yet confirmed on real data.
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
- **Not given** in these first three feature sets: natural gas or other
  fuel prices (added later, see the gas section), past prices, generator
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
- A true daily Algonquin gas series (paid: NGI, Platts) in place of the
  weekly one. See the gas section below.
- Day-ahead-appropriate inputs (previous-day information only).

## Adding natural gas prices (2026-10-09)

Tests the explanation above: that winter price spikes follow gas prices.

### The gas data

| Feature | What it is | Source |
|---|---|---|
| `gas_algonquin` | Algonquin Citygate spot price, $/MMBtu, the New England benchmark | Extracted from the text of EIA's Natural Gas Weekly Update |
| `gas_henry_hub` | Henry Hub daily spot price, $/MMBtu, the national benchmark | EIA `RNGWHHDd.xls` |

- **There is no free daily Algonquin series.** EIA's republished hub data
  stops in 2017; daily data is sold by NGI, Platts and others. EIA's weekly
  report quotes the Algonquin price for each Wednesday ("from $X last
  Wednesday to $Y yesterday"). `local/build_gas_prices.py` extracts these:
  **51 Wednesdays in 2024**, longest gap 14 days (no Dec 25 quote), with no
  disagreement where two reports quote the same day.
- **Weekly values are interpolated linearly to daily.** Anything that
  happened between Wednesdays is missed or smoothed, and each day's value
  uses the following Wednesday's quote.
- Henry Hub is a real daily series (251 trading days; weekends and holidays
  carry the last trading day).

Monthly mean Algonquin price: $6.23 in January, $4.07 in November, $7.67 in
December, and $1.62–$2.71 in every other month. Henry Hub stayed between
$1.49 and $4.02. Correlation of daily mean hub electricity price with
Algonquin is 0.66 (day-ahead); with Henry Hub, 0.36.

### Results with gas (day-ahead)

| Feature set | Chrono MAE | Chrono R² | Blocked MAE | Blocked R² |
|---|---|---|---|---|
| Calendar + weather + demand (no gas) | $24.21 | −0.23 | $9.86 | 0.56 |
| Calendar + weather + gas | $16.93 | 0.33 | $10.98 | 0.37 |
| Calendar + weather + demand + gas | $16.27 | 0.38 | $9.17 | 0.51 |
| Calendar + weather + demand + Henry Hub only | $23.26 | −0.13 | $7.52 | 0.74 |

"Gas" means both Algonquin and Henry Hub. Real-time shows the same pattern
on the chronological split (MAE $25.14 → $18.93, R² −0.24 → 0.22) and gets
worse on the blocked split (R² 0.53 → 0.13).

Chronological split, day-ahead, by test month:

| Month | Actual mean price | Predicted, no gas | Predicted, with gas | MAE no gas | MAE with gas |
|---|---|---|---|---|---|
| October | $36.0 | $31.0 | $27.5 | $9.2 | $8.9 |
| November | $39.5 | $31.7 | $37.6 | $11.1 | $10.0 |
| December | $87.9 | $35.8 | $65.0 | $52.2 | $29.8 |

### What this shows

- **Gas explains much of the December miss.** On the chronological split,
  adding gas moves the December prediction from $36 to $65 against an
  actual $88, cuts December error from $52 to $30, and takes R² from
  negative to 0.38. The models now beat the training-mean reference.
- **It is the regional price that matters.** Henry Hub alone does nothing
  on the chronological split.
- **December is still under-predicted by $23**, and cold and high-price
  hours remain the largest errors (top-10% prices under-predicted by $68,
  down from $103).
- **On the blocked split, the weekly gas series does not help overall, and
  hurts in places.** It fixes January (day-ahead MAE $40.6 → $13.4) but
  makes late November and late December worse ($5.0 → $12.6 and $34.4 →
  $48.2). Both are where the weekly series is least reliable: the late-
  November test days sit on an interpolated ramp from $2.84 to $9.82, and
  the late-December test days fall in the 14-day gap with no quote. This
  points at the coarseness of the weekly series, not at gas being
  irrelevant, but that reading has not been checked against daily data.
- **Henry Hub's blocked-split result (R² 0.74) should not be read as
  signal.** It does nothing on the chronological split. On the blocked
  split a slowly drifting daily series can act as a marker for which week
  a test day belongs to, next to training days from the same weeks.
- With one year and one run per model, differences of a few points of R²
  between feature sets on the blocked split are not reliable.

### Gas: how to reproduce

```bash
.venv/bin/python local/build_gas_prices.py --out-csv data/gas/gas_prices_daily_2024.csv
# then pass --gas-csv data/gas/gas_prices_daily_2024.csv to prepare_price_table.py
```

## Does training longer help? (2026-10-09)

Tested on the stripped-down day-ahead price model (calendar + weather +
actual demand, **no gas prices**), with early stopping switched off and the
error measured after every tree up to 10,000. Three seeds on the
chronological split; means shown.

| Trees | Chrono MAE | Chrono R² | December predicted (actual $87.9) | Blocked MAE |
|---|---|---|---|---|
| 20 (where the original run stopped) | $24.40 | −0.25 | $35.4 | n/a |
| 300 | $24.51 | −0.18 | $36.5 | $9.85 |
| 1,500 | $22.15 | −0.03 | $40.5 | $9.80 |
| 3,000 | $21.14 | 0.04 | $42.8 | $9.74 |
| 10,000 | $20.00 | 0.12 | $45.4 | $9.75 |

Blocked column is a single seed. Guessing the training average scores
$23.99 on the chronological split.

- **Longer training helps a little on the chronological split and not at
  all on the blocked split.** From 20 to 10,000 trees the chronological
  error falls from $24.4 to $20.0. The three seeds agree to within $0.4.
- **It does not close the gap.** December is still predicted at $45
  against $88, and the top 10% of prices are still under-predicted by $88.
  The improvement is slowing: the last 7,000 trees bought $1.1.
- **No overfitting on the test hours** was seen up to 10,000 trees, even
  though training error falls to $0.45.
- **The original early stopping was too impatient here.** It stopped at 18
  trees because the September validation block gave a flat, noisy signal.
  A fixed 3,000 trees would have been a better default for this split. The
  results tables above have NOT been rerun with that change.
- The missing information is not something more trees can supply: Jan-Aug
  2024 contains one cold-season price spike (January). More winters in the
  training data is the direct fix, which is the reason for adding years.

Figure: `results/price_dayahead_2024/training_length.png`
(`local/plot_training_length.py`).

### Blocked split in more detail (2026-10-09)

Same test, early stopping off, on the blocked split only.

Day-ahead price, mean of 3 seeds (seeds agree to within $0.5):

| Trees | Calendar + weather: MAE | R² | + demand: MAE | R² |
|---|---|---|---|---|
| 100 | $11.97 | 0.48 | $9.96 | 0.59 |
| 300 | $12.27 | 0.40 | $9.86 | 0.56 |
| 1,000 | $12.15 | 0.38 | $9.84 | 0.56 |
| 3,000 | $12.14 | 0.38 | $9.78 | 0.56 |
| 10,000 | $12.17 | 0.38 | $9.80 | 0.56 |

Demand, mean zone MAPE, mean of 2 seeds (seeds agree to within 0.3 points):

| Trees | Calendar | Calendar + weather | Vermont, calendar + weather |
|---|---|---|---|
| 100 | 14.4% | 9.6% | 26.8% |
| 300 | 12.5% | 6.3% | 13.6% |
| 1,000 | 11.4% | 5.6% | 10.6% |
| 3,000 | 11.1% | 5.5% | 10.1% |
| 10,000 | 10.8% | 5.5% | 10.1% |

- **Price: more trees do nothing** on this split. Error is flat from about
  300 trees; R² is slightly higher at 100 trees than later.
- **Demand: more trees help up to about 1,000, then stop.** The reported
  calendar + weather result (5.9%) came from a run that early-stopped at
  388 trees; around 1,000-3,000 trees it is 5.5%. Vermont improves from
  12.0% to about 10.1% but is still roughly double the other zones.
- **Calendar-only demand** keeps improving slowly (11.1% at 3,000, 10.8%
  at 10,000). It is the model that hit the 3,000-tree cap in the reported
  run.
- No overfitting on the test hours up to 10,000 trees in any of these.
- Taken with the chronological result above, early stopping with a
  100-tree patience has stopped too early in several runs. The headline
  tables have not been rerun; the plan is to use a fixed 3,000 trees when
  everything is rerun on the rebuilt data.

# 2020 on the working model: global grid, tokens in map order (2026-10-19)

The first year run the way the model needs: the whole globe (360 x 576), so
nothing is degraded by a regional crop, and the mask-unit order fixed, so
each saved token sits where the map says it does. All earlier embedding
results (sections above) used the cropped, degraded model and should be read
in that light. 2020 is a pandemic year, and everything here is within one
year.

Files: `prithvi_tokens_ne_2020*.npy` and `.json` (the sidecar says
`block_order: map`, `model_grid: 360 x 576`). There are 2,916 three-hourly
times; 12 are missing (about one and a half days, matching the day missing
from NASA's 2020 set). Every file was checked for complete length, NaNs and
blank rows.

## Setup

- **Zone area:** each zone is the 3 x 3 patches (about 3 x 3.75 degrees)
  around its reference point. Tokens are averaged over those 9 patches; the
  input, decoder and climatology fields are averaged over the matching 6 x 6
  grid cells.
- **Embedding time:** lead time 0 h. The model sees MERRA-2 at the target
  time and 6 h earlier, so this is still a nowcast with perfect knowledge.
- **Token sources:** `embed` (patch embedding, before any transformer
  block), encoder block 0, block 12, and the final encoder output.
- **Raw inputs:** the model's own 160 normalised input channels at the
  target time, for the same area. This is the "is the encoder adding
  anything?" baseline.
- **Principal components:** tokens and raw inputs are each reduced to 50
  components, fitted on training times only.
- **Script:** `local/analyze_global_year.py`. Outputs are in
  `results/global_2020/` (`temperature_probe.csv`, `demand_comparison.csv`).

## Temperature probe

Target: each zone's own MERRA-2 2 m temperature at its nearest grid cell
(the same number the weather baseline uses). Method: ridge regression, a
straight-line fit with a light penalty; one fit per zone. Train on days 1-18
of each month, test on day 22 onward. Scores are averaged over the 8 zones.

| Source | Dimensions | RMSE (K) | R² |
|---|---|---|---|
| Raw inputs, all 160 channels | 160 | **0.89** | 0.992 |
| Raw inputs, T2M channel only | 1 | 1.27 | 0.983 |
| Tokens: embed | 50 PCs | 1.31 | 0.983 |
| Tokens: block 0 | 50 PCs | 1.27 | 0.984 |
| Tokens: block 12 | 50 PCs | 1.74 | 0.971 |
| Tokens: final | 50 PCs | 1.74 | 0.971 |
| Decoder output, T2M channel | 1 | 1.32 | 0.982 |
| Climatology given to the model, T2M | 1 | 4.06 | 0.841 |

- **Local temperature is clearly in the tokens at every depth** (R² of 0.97
  or more, against 0.84 for climatology). The token layout is right: zone
  temperature reads out of the zone's own patches.
- **It fades with depth.** The early layers match the raw temperature
  channel. By block 12 the error is about 40% higher, and the final layer is
  no better. This fits an encoder that mixes in wider context as it goes.
- **The decoder reconstructs temperature almost exactly.** Its T2M differs
  from the true input by 0.007 sd on average, against 0.13 sd for
  climatology. As a temperature predictor it is about as good as the true
  temperature field itself. So the information is not lost inside the model;
  it is spread across many token directions, and 50 components of one
  layer do not capture all of it.
- **The raw input cube beats every token layer.** For recovering
  temperature, the encoder adds nothing over what it was given. Caveat: the
  inputs include the target time itself, so the raw cube nearly contains the
  answer. Most of its remaining 0.9 K is the difference between a 6 x 6-cell
  area mean and the single cell the target is taken from.

## Demand

Same CatBoost setup as before: one pooled model with `zone` as a
categorical feature, fixed 3,000 trees, learning rate 0.05, depth 6, seed 7.
The score is mean zone MAPE.

- **Blocked split:** train days 1-18 of each month, test day 22 onward.
- **Chronological split:** train January-August, test October-December.

| Features | Blocked | Vermont | Chronological | Vermont |
|---|---|---|---|---|
| Calendar | 10.47% | 17.0% | 11.75% | 19.3% |
| Calendar + weather | 4.93% | 8.4% | **8.36%** | 11.7% |
| Calendar + raw inputs | 5.39% | 7.8% | 10.40% | 15.7% |
| Calendar + tokens: block 0 | 5.49% | 7.9% | 15.00% | 16.8% |
| Calendar + tokens: block 12 | 5.70% | 8.1% | 16.34% | 18.7% |
| Calendar + tokens: final | 5.49% | 7.8% | 16.38% | 17.7% |
| Calendar + weather + tokens: final | **4.31%** | 6.7% | 10.44% | 14.3% |
| Calendar + weather + raw inputs | 4.32% | 6.4% | 9.46% | 12.6% |

The `embed` layer was still copying when this ran and is not in the demand
table.

- **Blocked split** (test days surrounded by training days from the same
  season): tokens alone come close to weather (5.5% against 4.9%). Adding
  them to weather helps (4.31%). But the raw input cube does the same
  (4.32%), so the gain comes from the extra weather information in the
  model's inputs, not from anything the encoder computes. Tokens alone are
  better than weather alone in Vermont (7.8% against 8.4%), and so are raw
  inputs.
- **Chronological split** (autumn, a season not in training): the
  high-dimensional features hurt badly. Tokens alone (15-16%) are worse
  than calendar alone (11.75%), and adding tokens to weather makes it worse
  (8.36% to 10.44%). Raw inputs also hurt, but less (9.46%). With 50
  components there are many ways for a tree model to latch on to
  season-specific patterns that do not carry over to new conditions.
  Six plain weather numbers generalise far better.
- **No sign so far that the encoder adds value over its own inputs** for
  this nowcast. The question that remains open is the one the encoder could
  win: a real forecast (`LEAD_TIME` > 0), where the raw inputs are stale and
  the model has to supply what happens next.

## Hub price, 2020, lead 0

Day-ahead hub price, one row per hour, no gas prices. Features are
system-level:
- weather: the eight zones' values averaged with fixed weights, each zone's
  share of annual demand;
- tokens and raw inputs: averaged over the whole 8 x 7 patch box, then 50
  principal components;
- decoder: the box mean of its 20 surface channels.

Same splits and fixed 3,000 trees as the demand models, one seed. Earlier
runs put seed-to-seed spread at about $0.5, so smaller differences are noise.

| Features | Blocked: MAE | R² | Chronological: MAE | R² |
|---|---|---|---|---|
| Calendar | $9.17 | -1.61 | $15.16 | -0.66 |
| Calendar + weather | $5.66 | 0.10 | $14.44 | -0.50 |
| Calendar + raw inputs | $5.79 | 0.14 | $12.83 | -0.33 |
| Calendar + tokens: embed | $5.38 | 0.32 | **$11.46** | -0.12 |
| Calendar + tokens: block 0 | $5.86 | 0.18 | $11.77 | -0.16 |
| Calendar + tokens: block 12 | $5.59 | 0.24 | $12.41 | -0.28 |
| Calendar + tokens: final | $5.41 | 0.29 | $12.42 | -0.27 |
| Calendar + decoder output | $6.03 | -0.22 | $14.01 | -0.45 |
| Calendar + weather + tokens: final | $5.21 | 0.30 | $13.04 | -0.34 |
| Calendar + weather + raw inputs | $5.01 | 0.35 | $13.01 | -0.32 |
| Calendar + weather + system demand | **$4.87** | 0.35 | $12.44 | -0.15 |

Test-period price: blocked mean $23.35 (sd $10.14); chronological
October-December mean $30.06 (sd $17.25).

- **2020 price is hard to model from weather.** Prices were low and
  variable (pandemic demand, cheap gas until a late-year rise). The best
  R² is 0.35, against 0.56 for the 2024 blocked split.
- **Chronological split:** every model does worse than simply predicting
  the test-period mean (R² below 0). The autumn rise in price is not in
  anything the models are given; it most likely follows gas prices, which
  are deliberately left out. Tokens are the least bad, but "less wrong"
  is not a useful result here.
- **Blocked split:** tokens alone beat weather alone (about $5.4 against
  $5.7; R² 0.3 against 0.1). Weather + raw inputs ($5.01) and weather +
  system demand ($4.87) are about equal, and all are within roughly $0.5
  of each other. This is no clear evidence that the encoder adds value
  over its inputs for price.

## Where to read the tokens from, 2020, lead 0

The earlier finding (pooled beats local, single patches worst) was measured
on the broken setup: regional crop, tokens in the wrong quadrant. Here it is
repeated on the working model.

Setup:
- final-layer tokens, 50 principal components fitted on training times;
- four ways to read the tokens for each zone;
- the same temperature probe and demand CatBoost as above, one seed.

Calendar + weather alone, for reference: 4.93% (blocked), 8.36%
(chronological).

| Readout | T2M RMSE (K) | Blocked: tokens | + weather | Chronological: tokens | + weather |
|---|---|---|---|---|---|
| Single nearest patch | 2.07 | 5.60% | **4.24%** | 14.99% | 10.77% |
| Zone 3 x 3 patches | **1.74** | **5.49%** | 4.31% | 16.38% | 10.44% |
| New England box mean | 1.97 | 7.34% | 4.48% | 17.92% | 9.23% |
| 30-60N, 90-50W pooled | 3.14 | 8.63% | 4.74% | 15.74% | **7.99%** |

- **On the working model the old ranking reverses** for temperature and the
  blocked split. A zone's own patches are best; the wide pooled vector is
  worst. The single nearest patch and the 3 x 3 window are about equal,
  within seed noise.
- **Chronological split** (new season): every readout of tokens hurts
  except the pooled one. Pooled + weather (7.99%) is slightly better than
  weather alone (8.36%), within about the size of seed noise. The pooled
  vector is identical across zones and changes slowly, so it gives a tree
  model less room to fit patterns that do not carry over.
- **Readout cost:** the zone's own patches (9 per zone, or 1) are the
  cheapest to save, and best wherever tokens help at all. Running the model
  still needs the whole globe as input.

## 24 hours ahead: 2020, lead 24 (2026-10-20)

The first real forecast test. Each sample is issued at time t (the model's
latest input) and describes t + 24 h. Every feature is lined up on that
target hour, so all Prithvi features and the "weather 24 h old" baseline are
a day old. "Weather at the target hour" uses the actual MERRA-2 values, so
it is a perfect-knowledge ceiling, not something available in practice.
Files: `prithvi_tokens_ne_2020_lead24*` (2,900 samples; checked complete,
no NaNs). Script: `analyze_global_year.py --lead 24`. Outputs:
`results/global_2020_lead24/`. Same setup as the lead-0 sections; one seed.

### Temperature at the target hour

| Source | RMSE (K) | R² |
|---|---|---|
| Decoder's 24 h forecast, T2M | **1.63** | 0.974 |
| Tokens: block 12 | 2.83 | 0.923 |
| Tokens: final | 3.00 | 0.913 |
| Tokens: block 0 | 3.05 | 0.910 |
| Raw inputs, 24 h old (160 channels) | 3.32 | 0.893 |
| Persistence: zone T2M 24 h earlier | 3.82 | 0.859 |
| Climatology for the target time | 4.06 | 0.841 |

- **The decoder is a real forecaster.** Its 24 h temperature error is
  less than half that of persistence.
- **The tokens carry forecast information beyond their inputs** (2.8-3.0 K
  against 3.3 K for the stale raw inputs). Unlike at lead 0, the middle
  layer is best. A linear readout gets much less out of them than the
  decoder does.

### Demand (mean zone MAPE)

| Features | Blocked | Chronological |
|---|---|---|
| Calendar | 11.07% | 11.78% |
| + weather at the target hour (ceiling) | 4.89% | **8.31%** |
| + weather 24 h old (no forecast) | 7.53% | 11.89% |
| + raw inputs, 24 h old | 6.51% | 12.21% |
| + tokens: block 0 / 12 / final | 6.72% / 6.60% / 6.40% | 14.26% / 14.62% / 14.53% |
| + decoder forecast, all 20 surface channels | 4.46% | 13.14% |
| + decoder forecast, the 5 baseline weather variables | 4.62% | 9.94% |
| + weather 24 h old + decoder, 20 channels | **4.27%** | 13.11% |
| + weather 24 h old + decoder, 5 weather variables | 4.39% | **9.50%** |
| + weather 24 h old + tokens: final | 5.82% | 10.52% |
| + weather 24 h old + raw inputs | 6.21% | 10.68% |

- **The decoder forecast is the first clear gain from the model.** With
  weather 24 h old as the honest baseline, adding the decoder's forecast of
  the same five weather variables cuts demand error from 7.53% to 4.39%
  (blocked) and from 11.89% to 9.50% (chronological). On the chronological
  split that recovers about two-thirds of the gap to the perfect-knowledge
  ceiling.
- On the blocked split the decoder even beats the ceiling (4.4-4.6%
  against 4.89%). A likely reason: the decoder features are averaged over
  the zone's 3 x 3 patch area, while the ceiling uses a single grid cell.
  Not tested.
- **Use only the decoder's weather variables.** With all 20 surface
  channels the chronological split fails (13.1%, worse than calendar
  alone). Slow seasonal channels such as leaf area and soil wetness let
  the trees memorise the training season. With 5 variables it generalises.
- **Tokens:** at lead 24 they beat stale raw inputs on both splits when
  combined with old weather (5.82% against 6.21%; 10.52% against 10.68%).
  The gain is small and within seed noise. The decoder output is far more
  useful than the tokens.

### Hub price, day-ahead

| Features | Blocked: MAE | R² | Chronological: MAE | R² |
|---|---|---|---|---|
| + weather at the target hour (ceiling) | $5.86 | 0.00 | $14.24 | -0.47 |
| + weather 24 h old | $7.24 | -0.53 | $14.50 | -0.59 |
| + weather 24 h old + decoder, 5 weather variables | $5.52 | 0.19 | $13.93 | -0.46 |
| + tokens: block 0 | $5.69 | 0.17 | **$11.58** | -0.16 |
| + weather at target + system demand | **$5.06** | 0.20 | $12.45 | -0.17 |

- Price is barely weather-driven in 2020 (see lead 0). The decoder forecast
  brings the stale-weather model to the level of the ceiling on the
  blocked split. Nothing works on the chronological split. Full table in
  `results/global_2020_lead24/run.log`.

### Fair ceiling: true weather over the same area (2026-10-21)

The decoder appeared to beat "perfect knowledge". That ceiling was six
MERRA-2 numbers at ONE grid cell, while the decoder features are averaged
over the zone's 3 x 3 patch area. The fair ceiling is the same five
variables, true values at the target hour, averaged over the same area in
the same units. Here it is taken from the lead-0 run's inputs.

| Demand features | Blocked | Chronological |
|---|---|---|
| Weather 24 h old (no forecast) | 7.53% | 11.89% |
| + decoder forecast, 5 variables, zone area | 4.39% | 9.50% |
| + true weather, 5 variables, zone area (fair ceiling) | **4.12%** | **8.82%** |
| True weather at one grid cell (earlier ceiling), alone | 4.89% | 8.31% |
| True weather over the zone area, alone | 4.39% | 10.07% |

- **The forecast does not beat the truth.** Against the fair ceiling the
  decoder recovers about 90% of the gap on the blocked split and 78% on the
  chronological split.
- The earlier "beats perfect knowledge" was an area-versus-point effect.
  Area-averaged weather is a better demand feature than one grid cell on the
  blocked split (4.39% against 4.89%), though not on the chronological split.

### Price with past prices (2026-10-21)

Day-ahead hub price, lead 24, adding past prices that are known when the
day-ahead market for the target day is bid: the hub price at the same hour
24 h, 48 h and 7 days earlier, and the mean of the 24 hours ending 24 h
earlier. No gas prices.

| Features | Blocked: MAE | R² | Chronological: MAE | R² |
|---|---|---|---|---|
| Calendar + weather 24 h old (no past price) | $7.24 | -0.53 | $14.50 | -0.59 |
| Calendar + past price | $6.73 | -0.29 | $10.74 | 0.04 |
| + weather 24 h old | $5.95 | -0.01 | $10.29 | 0.09 |
| + weather 24 h old + decoder weather | $4.60 | 0.42 | $9.55 | 0.23 |
| + weather 24 h old + tokens: final | $5.06 | 0.27 | **$8.53** | **0.34** |
| + weather at the target hour (ceiling) | **$4.55** | **0.45** | $10.93 | 0.08 |

- **Past prices rescue the chronological split.** Every model was below
  R² 0 without them; with them, all are above 0. They carry the
  autumn price level, which weather cannot.
- **The forecast matters for price too.** On the blocked split, decoder
  weather + past price ($4.60, R² 0.42) is almost exactly the
  perfect-knowledge ceiling ($4.55, R² 0.45). That is a large gain over
  stale weather + past price ($5.95).
- **Tokens + past price win the chronological split** ($8.53, R² 0.34),
  beating even the ceiling. This is the first case where tokens beat
  everything else. Treat with caution: one seed, one year, and a three-month
  test period. Possibly the tokens carry larger-scale weather (cold over a
  wider region drives gas demand and so prices) that zone weather does not.
  It needs the multi-year test.

### Price with gas prices added (2026-10-21)

Gas table for 2020 from `build_gas_prices.py`, same sources as 2024. The
Algonquin pattern now also matches "from a high of $X ... to a low of $Y"
(most 2020 reports): 52 quotes in 2020, at most 14 days apart. The 2024
table is unchanged.

Two versions:
- **Gas (known):** what a bidder has the morning before the day-ahead
  market clears. Henry Hub spot for the last trading day on or before D-2,
  and the last Algonquin quote on or before D-2, held flat.
- **Gas (same day):** day D's values, as in the 2024 models. The Algonquin
  interpolation uses the next weekly quote, so this is an upper bound.

Lead 24; "old weather" is weather 24 h before the target hour.

| Features (all with calendar + old weather) | Blocked: MAE | R² | Chronological: MAE | R² |
|---|---|---|---|---|
| + past price | $5.95 | -0.01 | $10.29 | 0.09 |
| + gas (known) | $6.88 | -0.55 | $13.34 | -0.36 |
| + past price + gas (known) | $5.60 | 0.10 | $9.64 | 0.14 |
| + decoder weather + past price + gas (known) | $4.61 | 0.39 | $9.49 | 0.22 |
| + tokens: final + past price + gas (known) | $4.65 | **0.44** | $8.66 | 0.33 |
| + past price + gas (same day; upper bound) | $4.92 | 0.39 | **$7.92** | **0.42** |
| Ceiling: weather at the target hour + past price + gas (known) | **$4.53** | 0.41 | $10.57 | 0.12 |

- **Known gas adds little once past prices are in** (about $0.35-0.65).
  Past prices already carry the price level.
- **Same-day gas helps much more on the chronological split** ($7.92,
  R² 0.42), but it partly uses the following week's gas price, so it is not
  achievable in practice.
- **With gas, the Prithvi features still matter.** Decoder or tokens +
  past price + known gas match the perfect-knowledge ceiling on the blocked
  split ($4.61 / $4.65 against $4.53). On the chronological split, tokens
  are best among achievable sets ($8.66, R² 0.33), much as without gas
  ($8.53).

### Caveats

- **One year, one seed, within-year splits.** It needs year-ahead
  training on 2021-2024.
- **The baseline is not a real forecast.** The fair practical comparison
  is against actual operational weather forecasts (for example archived NWS
  or GFS forecasts for the zone points), which beat persistence by a lot.
  The decoder may or may not beat those.
- **The inputs are themselves reanalysis.** MERRA-2 is published weeks
  late, so this is not yet a usable real-time forecasting setup; it shows
  what the model's representations can supply.

# GFS benchmark: real 24 h weather forecasts for demand (2026-10-22)

NOAA GFS 0.25-degree forecasts from the public AWS archive
(`src/gfs/download_gfs_forecasts.py`, `data/gfs/gfs_ne_YEAR.nc`). For each
3-hourly target time T: the run at T - 24 h (forecast hour 24) or T - 27 h
(hour 27), i.e. the latest run issued at least 24 h before T, as in the
lead-24 Prithvi setup. Script: `local/gfs_demand_benchmark.py`; outputs in
`results/gfs_benchmark_2023_2024/`.

Feature blocks per zone (T2M, QV2M, U10M, V10M, net shortwave, wind speed):
- **Weather (target):** MERRA-2 at the zone's grid cell at T. Perfect
  knowledge, single-cell ceiling.
- **Weather (issue):** the same at T - 24 h. Persistence.
- **GFS point:** the GFS forecast at the 0.25-degree cell nearest the zone
  point.
- **GFS area:** the GFS forecast averaged over the zone's 3 x 3 Prithvi patch
  window, the same area the decoder features use.

GFS shortwave is a 3-6 h average (DSWRF - USWRF), not an hourly value.
Every feature set is scored on the same hours (those with a GFS forecast).
At the time of this run, January 1 of each year was missing (since fixed).
CatBoost as elsewhere, 3,000 trees, one seed.

### 2 m temperature forecast check (zone grid cells, against MERRA-2)

| Year | GFS 24-27 h RMSE (K) | Bias (K) | Persistence RMSE (K) |
|---|---|---|---|
| 2023 | 2.09 | +0.18 | 3.77 |
| 2024 | 2.08 | +0.13 | 3.49 |

These are raw point-to-point errors. They are not comparable with the 2020
Prithvi decoder's 1.63 K, which came from a fitted linear readout of an
area average in a different year.

### Demand, mean zone MAPE

| Features | Blocked 2023 | Blocked 2024 | Chron. 2023 | Chron. 2024 | Year-ahead (train 2023, test 2024) |
|---|---|---|---|---|---|
| Calendar | 9.73% | 10.99% | 11.63% | 15.94% | 11.84% |
| + weather at T (ceiling, one cell) | 4.22% | 5.31% | 7.84% | 8.85% | 5.26% |
| + weather 24 h old | 6.79% | 7.44% | 10.77% | 11.38% | 7.80% |
| + GFS point | 4.44% | 5.31% | 8.42% | 10.29% | 5.40% |
| + GFS area | 4.18% | 5.15% | 8.48% | 9.18% | 5.01% |
| + weather 24 h old + GFS point | 4.17% | **4.85%** | 9.03% | 9.48% | 4.77% |
| + weather 24 h old + GFS area | **3.99%** | 4.86% | 8.80% | **8.31%** | **4.57%** |

- **A real forecast closes almost the whole gap.** In the year-ahead test,
  stale weather + GFS area (4.57%) beats the single-cell perfect-knowledge
  ceiling (5.26%). As with the Prithvi decoder, that is an area-versus-point
  effect plus the extra information in yesterday's weather, not a forecast
  beating the truth.
- **Area beats point** in most splits, as it did for MERRA-2 and the
  decoder.
- **For comparison with the 2020 Prithvi decoder** (different year, so
  only indicative): the share of the stale-to-ceiling gap closed is
  - Prithvi decoder, 2020: about 119% (blocked) and 67% (chronological);
  - GFS area: 2023 about 108% and 67%; 2024 about 122% and 121%.

  The two forecasts look broadly similar. A head-to-head needs lead-24
  Prithvi runs for 2021-2024.

# Head-to-head: Prithvi decoder against GFS, 2021-2024 (2026-10-26)

Lead-24 Prithvi runs for 2021-2024 on the global grid. Every year was
checked: Jan 1 to Dec 30 issue times, no gaps, no NaNs. The decoder's
normalised output is converted to physical units with one scale and offset
per variable. These were recovered from the data: the model's input pixel
on each zone's cell matches the weather table exactly (correlation 1.0).
Script: `local/gfs_demand_benchmark.py --prithvi-dir`; outputs in
`results/forecast_benchmark_2021_2024/`. Every forecast and feature set is
scored on the same times and hours.

### 2 m temperature 24 h ahead, against MERRA-2 (RMSE, K)

| | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| Point (zone grid cell): GFS | 2.10 | 2.12 | 2.09 | 2.08 |
| Point: Prithvi decoder | 1.31 | 1.40 | 1.34 | 1.33 |
| Point: persistence | 3.82 | 4.53 | 3.77 | 3.49 |
| Area (zone patch window): GFS | 1.04 | 1.01 | 1.02 | 1.00 |
| Area: Prithvi decoder | 0.84 | 0.92 | 0.86 | 0.86 |
| Area: persistence | 3.13 | 3.84 | 3.14 | 2.79 |

**This favours Prithvi.** The truth here is MERRA-2, which Prithvi was
trained on, starts from, and forecasts on the grid of. GFS starts from its
own analysis on a finer grid with different terrain, so it would differ from
MERRA-2 even at hour 0. That likely explains much of its single-cell error,
and fits the much smaller gap over the area. Fair versions are still to do:
1. GFS hour-0 analysis against MERRA-2, to measure the baseline
   disagreement.
2. Scoring both forecasts against independent station observations.

### Demand, year-ahead (mean zone MAPE)

| Features | Test 2023 (train 2021-22) | Test 2024 (train 2021-23) |
|---|---|---|
| Calendar | 11.27% | 11.55% |
| + weather 24 h old (no forecast) | 7.58% | 7.45% |
| + actual weather at T, one grid cell (ceiling) | 4.97% | 5.03% |
| + actual weather at T, zone area (ceiling) | 4.78% | 4.88% |
| + GFS point | 5.18% | 5.13% |
| + GFS area | 4.94% | 5.00% |
| + Prithvi decoder point | 4.86% | 5.04% |
| + Prithvi decoder area | 4.87% | 4.95% |
| + old weather + actual area weather (ceiling) | 4.49% | 4.72% |
| + old weather + GFS area | 4.54% | 4.73% |
| + old weather + Prithvi area | 4.67% | 4.92% |
| + old weather + GFS area + Prithvi area | **4.45%** | **4.66%** |

- **Both forecasts get demand to the perfect-knowledge ceiling.** With
  yesterday's weather, GFS reaches 4.54% / 4.73% and Prithvi 4.67% / 4.92%,
  against ceilings of 4.49% / 4.72%. Without a forecast it is 7.5%.
- **GFS is slightly ahead when combined with stale weather** (by 0.1-0.2
  points). Alone, they are level. These gaps are about the size of
  seed-to-seed noise seen earlier (0.1-0.3 points).
- **Both together are best in both years**, a little below the ceiling:
  the two forecasts carry partly different information. The gain over GFS
  alone is small (0.07-0.09 points).
- **Prithvi's better temperature score against MERRA-2 does not carry over
  to demand.** Both forecasts are already close to the ceiling, so there is
  little room to separate them, and the temperature comparison is biased
  (above).
- **The starting point favours Prithvi.** Each Prithvi forecast starts
  from MERRA-2, a reanalysis made afterwards with hindsight and more
  observations than GFS had when it ran in real time. So "matches GFS" is an
  upper bound on what Prithvi would do in operation. MERRA-2 is also
  published weeks late, so it cannot be used in real time.
- **The fair test:** start Prithvi from an operational analysis available
  in real time, such as NASA's GEOS-FP (same model family and file layout as
  MERRA-2; archive 2014 to now, 0.25 x 0.3125 degree, which contains the
  MERRA-2 grid points exactly) or the GFS analysis. If it still matches
  GFS then, that is the strong result.

### Demand with lagged load (2026-10-26)

Every feature set also gets the zone's own demand 24 h and 168 h before the
target hour. Both are known when a day-ahead forecast is made.
`gfs_demand_benchmark.py --load-lags`; output:
`results/forecast_benchmark_2021_2024/demand_with_load_lags.csv`. Same test
hours as above. The first week of 2021 drops out of training (no lags).

| Features (all with calendar) | Test 2023 | Test 2024 | Without lags, 2023 / 2024 |
|---|---|---|---|
| + load lags | 6.84% | 6.95% | 11.27% / 11.55% (calendar) |
| + load lags + weather 24 h old | 6.43% | 6.57% | 7.58% / 7.45% |
| + load lags + old weather + GFS area | 3.79% | 3.96% | 4.54% / 4.73% |
| + load lags + old weather + Prithvi area | 4.01% | 4.13% | 4.67% / 4.92% |
| + load lags + old weather + GFS + Prithvi | 3.75% | **3.87%** | 4.45% / 4.66% |
| + load lags + old weather + actual area weather (ceiling) | **3.74%** | **3.87%** | 4.49% / 4.72% |

- **Lagged load helps every model** by about 0.7-0.8 points, and alone
  nearly halves the calendar-only error. Yesterday's weather adds little
  once yesterday's load is known (6.84% to 6.43%).
- **A forecast is still essential.** Adding GFS or Prithvi on top of the
  lags cuts error from about 6.5% to about 4%.
- **The ranking is unchanged.** With lags, GFS is at the ceiling (within
  0.05-0.09 points) and Prithvi is about 0.2 points behind GFS. Both
  together equal the ceiling.

# Price, 2021-2024: heat rate and the gas-free test (2026-10-26)

Day-ahead hub price, 24 h ahead, year-ahead folds (test 2023 trained on
2021-22; test 2024 trained on 2021-23). Script: `local/price_benchmark.py`;
outputs in `results/price_benchmark_2021_2024/`.
- **Features:** every set has calendar + past values of its target (24 h,
  48 h and 7 days before, and the previous day's mean). Weather blocks are
  zone values weighted by each zone's share of demand. Tokens are the
  New England box mean, 50 principal components fitted on training times.
  No load lags.
- **Scoring:** MAE in $/MWh and R² on the price, same hours for every set.
  One seed.

**A. Implied heat rate.** The model predicts LMP / that day's Algonquin
price; the price is the prediction x that day's Algonquin. Gas is a
normaliser only, never a feature. Median heat rate 11.3 MMBtu/MWh. Caveat:
our daily Algonquin is interpolated between weekly EIA quotes, so it is a
normaliser, not something known in advance.

**B. No gas at all (the scientific test).** The model predicts LMP from past
LMP and weather only. (A first run leaked the target through a column-name
match, giving R² 1.00; fixed, and a check now stops the script if it
recurs.)

| Features (+ calendar + past values) | A 2023: MAE | R² | A 2024: MAE | R² | B 2023: MAE | R² | B 2024: MAE | R² |
|---|---|---|---|---|---|---|---|---|
| Nothing else | $11.93 | 0.28 | $9.21 | 0.65 | $14.27 | 0.12 | $10.59 | 0.59 |
| + weather 24 h old | $10.98 | 0.44 | $8.87 | 0.66 | $13.10 | 0.26 | $9.92 | 0.65 |
| + old weather + GFS area | $9.47 | 0.58 | $7.37 | 0.74 | $9.62 | 0.62 | $7.55 | 0.79 |
| + old weather + Prithvi decoder | $9.36 | 0.59 | $7.44 | 0.74 | $9.61 | 0.62 | $7.72 | 0.79 |
| + old weather + GFS + Prithvi decoder | **$9.21** | **0.61** | **$7.29** | **0.75** | $9.24 | 0.65 | $7.58 | 0.80 |
| + old weather + Prithvi tokens | $9.79 | 0.54 | $8.17 | 0.69 | $9.36 | 0.61 | $7.69 | 0.78 |
| + old weather + GFS + Prithvi tokens | $9.30 | 0.59 | $7.58 | 0.71 | **$8.42** | **0.70** | **$6.90** | **0.82** |
| + old weather + actual area weather at T (ceiling) | $9.37 | 0.59 | $7.40 | 0.75 | $9.76 | 0.64 | $8.44 | 0.76 |

- **A weather forecast matters for price** in both versions. It cuts error
  by 15-17% in A and 24-27% in B, relative to stale weather.
- **GFS and the Prithvi decoder are level** in both versions (within
  about $0.15).
- **The tokens behave differently from the decoder:**
  - In A (gas level removed), they add nothing, or slightly hurt.
  - In B (no gas information), GFS + tokens is clearly best in both years:
    $8.42 against $9.62 for GFS alone, and $6.90 against $7.55. That is
    9-12% lower error, and beats the local perfect-knowledge ceiling.
  - So the tokens carry something about price beyond local weather, and
    that something largely disappears once the gas level is divided out.
    That fits the hypothesis that they anticipate weather-driven gas
    conditions (for example wider cold spells, which push up gas demand
    and prices) before past prices show them.
- **Not yet shown:**
  - Whether the token gain is concentrated on gas-spike days.
  - Whether it is just a proxy for season or gas regime. A test with
    calendar-by-season interactions, or with gas included, would help.
  - One seed; the gains are larger than seed noise seen before, but are not
    tested here.

## Price v2: four normalisers, hard days, bootstrap (2026-10-26)

`local/price_benchmark.py` (outputs `results/price_benchmark_v2/`:
`price_summary.csv`, `price_predictions.parquet`). Same year-ahead folds and
feature sets as above, plus "+ GFS + decoder + tokens". Each model predicts
LMP / N; the price is the prediction x N, and errors are on the price.

| Version | Normaliser N, known when the day-ahead market is bid |
|---|---|
| A (main) | Henry Hub spot, last trading day on or before D-1 |
| A' | last published Algonquin quote dated on or before D-2, carried forward, never interpolated |
| B-norm | mean day-ahead hub LMP over the 7 days ending 24 h before the target hour (gas-free) |
| B | none (raw LMP, gas-free) |

**Hard days** are defined from the actual temperature over the zone areas
(system-weighted), with thresholds from the training years only.
- **Cold snap:** daily mean at or below the 5th percentile of training
  days.
- **Heat wave:** daily maximum at or above the 95th percentile.

Test hours: 96 cold-snap and 120 heat-wave hours in 2023 (4 and 5 days);
288 and 336 in 2024.

**Intervals:** 95% intervals come from resampling calendar weeks (2,000
draws) of paired hourly differences in absolute error, against
"+ old weather + GFS". Negative means better than GFS.

### Overall MAE ($/MWh) and the difference from GFS

| Features (+ calendar + past values) | A 2023 | A 2024 | A' 2023 | A' 2024 | B-norm 2023 | B-norm 2024 | B 2023 | B 2024 |
|---|---|---|---|---|---|---|---|---|
| + old weather | 10.37 | 9.98 | 13.27 | 10.08 | 11.79 | 9.38 | 13.11 | 9.80 |
| + old weather + GFS (reference) | 8.55 | 8.30 | 10.72 | 8.84 | 8.78 | 7.93 | 9.54 | 7.64 |
| + Prithvi decoder instead of GFS | 8.70 | 8.11 | 11.05 | 8.87 | 8.90 | 8.03 | 9.64 | 7.82 |
| + GFS + decoder | 8.50 | 8.13* | 10.92 | 8.71 | 8.67 | 7.84 | 9.19* | 7.44 |
| + GFS + tokens | 8.10* | 8.43 | 10.98 | 8.85 | 8.55 | 7.87 | 8.46* | 6.94* |
| + GFS + decoder + tokens | 8.13* | 8.38 | 11.04 | 8.93 | 8.56 | 7.65 | 8.35* | 7.01* |
| + actual area weather at T | 8.58 | 8.36 | 10.80 | 8.71 | 8.73 | 7.94 | 9.91 | 8.37 |

\* 95% interval of the difference from GFS excludes zero (better than GFS).

- **A weather forecast matters for price in every version.** GFS cuts MAE
  by $1.5-3.6 relative to stale weather. The Prithvi decoder alone is level
  with GFS (no interval excludes zero in its favour).
- **The token gain is mostly a regime effect.** On raw LMP (B), GFS +
  tokens is better than GFS in both years ($1.08 [0.52, 1.66] and $0.70
  [0.09, 1.44]). Once the price level is normalised, the gain shrinks and is
  no longer significant:
  - B-norm: $0.24 and $0.05;
  - A' (Algonquin): none;
  - A (Henry Hub): $0.45 [0.17, 0.72] in 2023 only.

  (Revised below: with gas given explicitly as a feature, the gain survives.)
  So most of what the tokens add to raw LMP is tracking the price level
  (season, gas regime), not anticipating individual weather-driven spikes.
  A small part may survive under the Henry Hub normaliser (2023), where the
  New England premium is left in.
- **Combining GFS + decoder** is slightly better than GFS in most cases,
  significant in two of eight.
- **Forecasts can beat "perfect knowledge" for price.** On B, the actual
  area weather is worse than GFS ($0.37 and $0.73 higher MAE). Day-ahead
  prices are set from the forecasts the market had, not from the weather
  that happened, so a forecast is the more relevant feature.
- **Hard days:** cold-snap MAE is 3-10 times the overall MAE (for example
  about $95 in 2023 for A, driven by the February 3-4, 2023 cold outbreak).
  No feature set reliably reduces it. The 2023 test year has only 4
  cold-snap days, too few for the week bootstrap to be meaningful (its
  intervals there are artificially narrow). On 2024 cold snaps (12 days)
  intervals are wide and all include zero, except "+ actual area weather"
  for A (worse). Heat-wave days: the decoder alone is often worse than GFS.
- **Top 5% of price hours:** GFS and GFS + decoder are best or tied in
  most versions; tokens tend to be worse there.

## Price: known gas as a feature, and a raw-input control (2026-10-26)

Raw day-ahead hub LMP, 24 h ahead, year-ahead folds (test 2023 trained on
2021-22; test 2024 trained on 2021-23), CatBoost as before (3,000 trees,
depth 6, learning rate 0.05, one seed). Every feature set has calendar
features (hour, day of week, day of year, month, weekend, federal holiday;
no year), past prices (24 h, 48 h, 7 days, mean of the 24 h ending 24 h
earlier) and, new here, **gas prices known at bid time as features**: Henry
Hub for the last trading day on or before D-1, optionally also the last
Algonquin quote dated on or before D-2 (carried forward). Errors in $/MWh;
"Δ vs GFS" is the paired difference from "+ old weather + GFS" with a 95%
week-block bootstrap interval (negative = better). Script:
`local/price_benchmark.py --targets lmp --gas-features ... [--input-control]`;
outputs `results/price_known_gas/`.

This replaces normalisation as the test of whether the tokens only encode
the fuel regime: the model is told the gas price directly.

MAE, with Henry Hub known (2023 / 2024):

| Feature set (all + calendar + past prices + Henry Hub) | MAE | Δ vs GFS 2023 | Δ vs GFS 2024 |
|---|---|---|---|
| no weather | 12.41 / 12.63 | +4.01 | +5.02 |
| + old weather | 11.10 / 12.01 | +2.69 | +4.41 |
| + old weather + GFS (reference) | 8.41 / 7.60 | | |
| + old weather + decoder alone | 8.59 / 7.84 | +0.18 [-0.08, +0.47] | +0.23 [-0.05, +0.56] |
| + old weather + tokens alone | 8.64 / 7.47 | +0.23 [-0.23, +0.72] | -0.13 [-0.65, +0.42] |
| + old weather + decoder + tokens | 7.86 / 6.99 | -0.55 [-1.00, -0.11] | -0.62 [-0.95, -0.25] |
| + GFS + decoder | 8.21 / 7.65 | -0.20 [-0.40, -0.01] | +0.05 [-0.22, +0.30] |
| + GFS + tokens | 7.55 / 6.74 | **-0.86 [-1.33, -0.45]** | **-0.86 [-1.23, -0.52]** |
| + GFS + decoder + tokens | 7.47 / 6.80 | -0.94 [-1.43, -0.50] | -0.80 [-1.11, -0.50] |
| + old weather + actual area weather | 8.69 / 8.03 | +0.28 [+0.02, +0.51] | +0.42 [+0.10, +0.78] |

With Henry Hub and Algonquin both known, the token gain on top of GFS is
-0.80 [-1.34, -0.33] (2023) and -0.70 [-1.10, -0.34] (2024).

**The token gain survives explicit gas.** Without gas it was -1.08 / -0.70;
with Henry Hub it is -0.86 / -0.86 (about 10-11% of MAE). So the tokens are
not just a gas-regime proxy, and day of year and month are already in the
calendar, so they are not just "winter" either.

**Control: Prithvi's own MERRA-2 input fields.** Same construction as the
tokens (all 160 input channels at issue time, averaged over the New England
box, 50 principal components fitted on training times), with Henry Hub known:

| | 2023 | 2024 |
|---|---|---|
| + GFS + MERRA-2 fields, Δ vs GFS | -0.76 [-1.24, -0.27] | -0.39 [-0.68, -0.10] |
| + GFS + tokens, Δ vs GFS | -0.86 | -0.86 |
| tokens vs MERRA-2 fields (both + GFS) | -0.09 [-0.43, +0.24] | **-0.48 [-0.69, -0.26]** |
| + tokens on top of GFS + MERRA-2 fields | -0.20 [-0.50, +0.05] | **-0.40 [-0.62, -0.18]** |

- In 2023, a broad snapshot of the current atmosphere (the raw analysis)
  gives nearly all of the token gain. That is information the market
  already has at bid time; it is not something only Prithvi's processing
  adds.
- In 2024, the tokens are significantly better than the raw fields and add
  to them. So a Prithvi-specific contribution exists in one of two years.
- On their own (with old weather, no GFS), neither the decoder nor the
  tokens beat GFS; the gains come from combining them with GFS.
- Revises the v2 reading ("mostly a price-level effect"): explicit gas does
  not remove the gain; normalising the target changed the learning
  problem rather than isolating the regime.

Caveats: one seed; 50 extra features versus 6; the MERRA-2 analysis at
issue time is a reanalysis, not an operational analysis (see the GEOS-FP
note above).

## Price: token readout, PCA or not, pooled or spatial (2026-10-26)

Same setting as the known-gas section (raw LMP, Henry Hub known as a
feature, year-ahead folds, one seed). Four token readouts, each with old
weather, alone and on top of GFS. Script: `local/price_benchmark.py
--targets lmp --gas-features henryhub --token-variants`; outputs
`results/price_known_gas/token_variants/`.

- pooled full: final-layer tokens averaged over the 8 x 7 box, all 2,560
  dimensions, no PCA.
- pooled PCA-50: the readout used so far. 50 standardised principal
  components fitted on training times; they keep 76% (2023 fold) / 75%
  (2024 fold) of the variance.
- spatial k10 / k25: every patch kept (56), each patch's 2,560-d token
  projected on k components of ONE basis shared by all patches (fitted on
  training times and patches): 560 or 1,400 features. 10 components keep
  24% and 25 keep 42% of the per-patch variance; individual patches are
  much less compressible than the box average.

MAE ($/MWh) and Δ vs "+ old weather + GFS" (8.41 / 7.60) with 95%
week-bootstrap interval:

| Token readout | + old weather, no GFS: MAE 2023 / 2024 | + GFS: MAE 2023 / 2024 | Δ vs GFS 2023 | Δ vs GFS 2024 |
|---|---|---|---|---|
| pooled full (2,560) | 8.50 / 7.53 | 7.80 / 7.18 | -0.60 [-1.29, -0.03] | -0.43 [-0.86, +0.06] |
| pooled PCA-50 | 8.64 / 7.47 | **7.55 / 6.74** | **-0.86 [-1.33, -0.45]** | **-0.86 [-1.23, -0.52]** |
| spatial k10 (560) | 8.51 / 7.60 | 7.57 / 7.02 | -0.83 [-1.39, -0.37] | -0.59 [-0.93, -0.25] |
| spatial k25 (1,400) | 8.66 / 7.57 | 7.68 / 7.13 | -0.72 [-1.31, -0.21] | -0.48 [-0.82, -0.11] |

- Does Prithvi help? Yes, on top of GFS, with every readout; never alone
  (all "no GFS" rows are within noise of GFS).
- Does PCA throw away useful information? No sign of it: the full
  2,560-d block is worse than PCA-50 in both years. With about 17-26
  thousand training hours, more raw dimensions dilute rather than inform.
- Does keeping the geography help? No: spatial readouts are between full
  and PCA-50, and more components per patch is worse (k25 < k10).
- Caveat: one seed, one CatBoost configuration; the ranking among
  readouts (differences of $0.0-0.4) is within the intervals' width.

## Demand: hard days, tokens, and where the best model struggles (2026-10-26)

Pooled zonal demand, 24 h ahead, year-ahead folds as above, no load lags.
Calendar features: hour, day of week, day of year, month, weekend, federal
holiday, zone. New block: **Prithvi tokens per zone** (final layer, the
zone's 3 x 3 patch window, 50 principal components fitted on training
times). Hard days defined from the actual area temperature weighted by
demand share, with training-year thresholds (cold snap: daily mean at or
below the 5th percentile, 268.5 K / 269.9 K; heat wave: daily maximum at or
above the 95th, 299.5 K / 299.0 K). 2023 has 4 cold-snap and 6 heat-wave
test days; 2024 has 14 and 15. Top 5% = the 5% of test hours with the
highest system demand. MAPE (%), mean over zones. Script:
`local/gfs_demand_benchmark.py --hard-days`; outputs
`results/demand_hard_days/` (predictions, summary, daily MAPE).

| Feature set (all + calendar + old weather) | MAPE 2023 / 2024 | Δ vs GFS 2023 | Δ vs GFS 2024 | cold snaps 2023 / 2024 | heat waves 2023 / 2024 | top 5% 2023 / 2024 |
|---|---|---|---|---|---|---|
| calendar only (no weather) | 11.27 / 11.55 | | | 10.79 / 11.47 | 18.15 / 16.07 | 14.25 / 15.55 |
| old weather only | 7.58 / 7.45 | +3.04 | +2.72 | 9.78 / 7.66 | 15.66 / 8.90 | 11.78 / 10.01 |
| + GFS (reference) | 4.54 / 4.73 | | | 6.20 / 4.65 | 5.17 / 4.95 | 5.95 / 5.17 |
| + decoder alone | 4.67 / 4.92 | +0.13 [+0.03, +0.23] | +0.19 [+0.08, +0.31] | 5.98 / 4.78 | 5.99 / 5.63 | 6.22 / 5.98 |
| + tokens alone | 5.77 / 6.22 | +1.23 [+1.04, +1.41] | +1.49 [+1.27, +1.73] | 6.50 / 5.88 | 6.48 / 6.39 | 7.08 / 7.40 |
| + decoder + tokens | 4.78 / 4.94 | +0.24 | +0.20 | 5.16 / 5.03 | 5.48 / 5.15 | 6.14 / 5.54 |
| + GFS + decoder | **4.45 / 4.66** | -0.09 [-0.16, -0.03] | -0.08 [-0.12, -0.03] | 6.04 / 4.67 | 4.99 / 5.00 | 5.80 / 5.32 |
| + GFS + tokens | 4.72 / 4.77 | +0.18 [+0.07, +0.30] | +0.03 [-0.10, +0.17] | 4.91 / 4.82 | 5.05 / 4.80 | 5.88 / 4.98 |
| + GFS + decoder + tokens | 4.59 / 4.67 | +0.05 | -0.06 | 4.89 / 4.84 | 5.07 / 4.78 | 5.77 / 4.91 |
| + actual area weather (ceiling) | 4.49 / 4.72 | -0.05 [-0.15, +0.05] | -0.01 [-0.10, +0.08] | 6.06 / 4.09 | 4.84 / 4.60 | 5.64 / 5.39 |

- For demand the decoder is the useful Prithvi output; tokens alone are
  much worse (+1.2 to +1.5 points), and add nothing on top of GFS overall.
- GFS + decoder is the best model in both years, by a small but
  significant 0.08-0.09 points, essentially at the actual-weather ceiling.
- Cold snaps: tokens on top of GFS cut 2023 cold-snap MAPE from 6.2% to
  4.9% (interval excludes zero), but that is 4 days (the February 3-4, 2023
  outbreak), too few for the week bootstrap to be trusted; in 2024 (14
  days) there is no gain. Heat waves and the top 5% of hours: no reliable
  Prithvi gain over GFS.

**Where the best model struggles.** The 12 worst days of GFS + decoder
(daily MAPE 8-14%, against about 4.5% overall) are mostly not weather days:

- 2023: Memorial Day (14.0%), Christmas, Thanksgiving and the day after
  (not flagged as a holiday), July 3-4, plus summer weekends and December
  18 (the wind storm with widespread outages in Maine). Only one is a cold
  snap (February 4).
- 2024: spring days in April-May and summer days in June, Juneteenth, one
  cold-snap Sunday in December.
- The actual-weather model is just as bad on these days (for example 11.7%
  on Memorial Day 2023), so better weather input cannot fix them. They are
  calendar effects (holidays, bridge days, weekends) and events such as
  outages; plausibly behind-the-meter solar on clear spring days in 2024
  (not tested).
- Does Prithvi help on the hardest days? Picking the worst 5% of days with
  the actual-weather model (neutral between GFS and Prithvi): 2023 GFS
  8.28%, + decoder 8.32%, + tokens 8.71%; 2024 GFS 7.26%, + decoder 7.16%,
  + tokens 6.92%. No consistent help. (Picking days by GFS's own error
  favours any other model by regression to the mean, so it is not used.)
- Cheap next step suggested by this: a bridge-day / holiday-adjacent
  feature (day after Thanksgiving, July 3, December 24/26).

## Still to do (2020 onward)

- Run 2021-2024 the same way and repeat with training on earlier years
  (year-ahead), which is the setting that matters.
- Forecast test: embeddings with `LEAD_TIME=24`, compared against raw
  inputs from 24 h earlier and against actual weather forecasts.
- Fewer components (for example 5-10) for tokens and raw inputs on the
  chronological split, to check whether the failure is simply too many
  features.
