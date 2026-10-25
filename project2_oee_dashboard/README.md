# Production Efficiency (OEE) & Downtime Analytics Dashboard

**Live app:** _add your Streamlit Community Cloud link here_  |  **Screenshots:** put PNGs in `docs/screenshots/` and embed them here

![Plant overview](docs/screenshots/01_overview.png)

> **Synthetic data for portfolio purposes.** Every file in `data_production/` is synthetic (seed 42, year 2025). Plant and line names are generic and are **not** real Danone plants. OEE benchmark bands are a common rule of thumb, not a company target. Money values and the CO2e grid factor are illustrative assumptions, editable in the sidebar.

## Problem statement
A multi-line beverage / dairy / nutrition manufacturer runs 6 lines in 2 plants over 3 shifts. Management asks three questions:
**where do we lose the most, why, and what do we fix first?** This dashboard answers them with Pareto thinking, an OEE loss tree, root-cause hints, early-warning analytics and a ranked list of improvements quantified in units, hours and IDR.

## Data (`data_production/`)
| File | Grain |
|---|---|
| `shift_production.csv` | line x shift (5,807 rows) - minutes, units, crew, energy, water, waste, CO2e |
| `downtime_events.csv` | stop event (38,174) with loss category, equipment, reason |
| `quality_defects.csv` | shift x defect type |
| `maintenance_log.csv` | work order (2,708): preventive / corrective, cost inputs |
| `line_master.csv`, `plant_master.csv`, `line_sku_rates.csv` | masters |

The CSVs are used as delivered (not modified). The generator script that produced them (`generate_production_dataset.py`) is not part of this repo snapshot; add it at the root if you want to regenerate (seed 42).

## Methodology
```
Availability = operating_time / planned_production_time
Performance  = total_units / (operating_time x ideal_rate)
Quality      = good_units / total_units
OEE          = A x P x Q
```
* **Weighted aggregation:** for any period / line / filter, numerators and denominators are **summed first, then divided**. A simple average of shift percentages over-weights small shifts; `tests/test_oee.py` shows both (example: naive 60.0% vs weighted 27.3%).
* **Loss tree** (minutes): planned time -> 7 availability losses (logged events; they reconcile exactly with planned - operating time) -> minor stoppages (logged) -> reduced speed (residual speed loss) -> rejects (`reject_units / ideal_rate`) -> fully productive time (`good_units / ideal_rate`). A test asserts planned - losses = productive time.
* **MTBF** = operating hours of the line / breakdowns on that equipment; **MTTR** = mean breakdown duration.
* **Shift effect:** OEE regressed on line fixed effects + shift (robust HC3 errors) so S3 vs S1 is a *within-line* difference; cross-checked with a bootstrap on matched line-weeks.
* **Before/after (L02):** weighted-OEE difference with a bootstrap CI that resamples whole weeks (shifts of a line are autocorrelated).
* **SPC:** p-chart with 3-sigma limits and run rules (rule 1 beyond limits, rule 2 eight in a row). With 50-100k units per shift a classic p-chart flags most shifts on 5 of the 6 lines (64-88%), so the default is the **Laney p'** chart, which scales limits by the real shift-to-shift variation.
* **Early warning:** weekly breakdown minutes vs `median + k x 1.4826 x MAD` of a trailing baseline made of non-alarm weeks (so a long problem cannot become the new normal); IsolationForest on [breakdown min, count, reject rate, speed ratio] as a cross-check; risk score = 100 x (0.4 freq rank + 0.3 severity rank + 0.3 trend rank); optional logistic regression with a strict chronological split.
* **Impact:** recovered minutes x line's realised good units per operating minute x volume-weighted contribution margin (assumption). Annualised from the selected period.

## Pages
| Page | Decision question |
|---|---|
| 1 Plant Overview | Which line / period needs attention first? (KPIs with delta, benchmark bands, trend, heatmap, auto "So what?") |
| 2 OEE Loss Analysis | Where is time lost? Waterfall, Pareto drill-down category > equipment > reason, MTBF/MTTR, line/shift/month comparison |
| 3 Downtime Deep Dive | Which stops matter, when, and what is controllable? Frequency vs severity, time-of-day heatmap, changeover + SMED what-if, start-up-after-idle, shift effect |
| 4 Quality | Which defects / conditions drive rejects? Pareto, mix, SPC p-chart + flagged-shift table, cost of poor quality |
| 5 Maintenance & Early Warning | Can we see a failing line early? Preventive vs corrective cost, weekly alarms + lead time, risk watch-list, optional model |
| 6 Sustainability | What do losses cost in energy / water / waste / CO2e, and what do we save by fixing the top loss? |
| 7 Improvement Opportunities | What do we fix first and what is it worth? Ranked actions, effort/impact quadrant, L02 before/after |

All assumptions (margin per SKU, unit cost, labour rate, grid factor, idle energy/water shares, improvement shares, effort scores) live in one **Assumptions** expander in the sidebar and flow through every calculation.

## Key findings (computed from this data, default assumptions)
1. Plant-wide OEE is **68.8%** (A 85.1%, P 83.0%, Q 97.4%). The largest loss bucket is **reduced speed** (11.1% of planned time), then breakdowns (9.1%); only 69.9% of planned time is fully productive.
2. L02's Filler caused **420 hours** of breakdown in Apr-Sep (~70 h/month vs ~2 h/month after the fix). L02 OEE recovered from **57.0% to 70.1%** (+13.1 pts, 95% CI 10.8-15.2).
3. L05 changeovers average **92 min** (vs ~51 on PET lines). A 30% SMED cut recovers ~98 h and adds ~0.59 M units/yr (~IDR 1.14 bn).
4. Night-shift OEE is **5.3 pts below** day shifts (-5.2 pts within line, 95% CI -6.0 to -4.4): lower performance and availability, more rejects, and 412 waiting-operator stops vs 222 / 235 on S1 / S2.
5. The robust weekly alarm flagged L02 in the week of 7 Apr 2025, **10 weeks before** its worst 4-week window. Other lines raised 17 alarm-weeks in total (the false-alarm cost).

More detail (L06 shortages, start-up after idle, FRM-900 learning curve, rainy season, ranked opportunities): see [`FINDINGS.md`](FINDINGS.md).

## Project structure
```
app.py                 # navigation + shared sidebar filters/assumptions + footer
pages/p1..p7_*.py      # one file per page
src/data.py            # loading, joins, derived columns (idle-before, SKU change, loss minutes)
src/oee.py             # weighted OEE, loss tree
src/pareto.py          # Pareto, drill-down, MTBF/MTTR
src/spc.py             # p-chart / Laney p', run rules
src/stats.py           # regression with robust SE, bootstrap, z-tests
src/anomaly.py         # MAD alarm, IsolationForest, risk score, breakdown model
src/impact.py          # units / hours / IDR / CO2e, opportunity table
src/viz.py, ui.py      # chart theme, filters, assumptions panel
tests/                 # unit tests (OEE maths, loss-tree reconciliation, Pareto, SPC, stats...)
scripts/build_findings.py   # regenerates FINDINGS.md from the data
data_production/       # synthetic CSVs
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Tests: `pip install -r requirements-dev.txt && python -m pytest -q` (or `python tests/run_tests.py` without pytest). Regenerate findings: `python scripts/build_findings.py`.

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub (CSVs are ~8 MB and are committed), then in share.streamlit.io choose the repo, branch `main`, main file `app.py`. Data loads with `@st.cache_data`.

## Limitations
* Synthetic data: effects were embedded by a generator, so significance tests demonstrate *method*, not real-world truth.
* L02 before/after has no control line; seasonality or volume changes could contribute.
* IDR impacts depend on illustrative contribution margins; levers overlap, so the total is an upper bound.
* Energy / water models use assumed idle shares (a regression cross-check is shown); CO2e uses an assumed grid factor of 0.85 kg/kWh.
* Next-shift breakdown prediction is weak (precision close to the base rate); the weekly alarm and watch-list are more useful.
* Work orders cannot be filtered by shift / SKU.

## What I would do with real MES / SCADA data
* **Real-time OEE** from PLC counters instead of shift summaries, with a live loss tree per line.
* **Automatic reason codes** from machine states and alarms, replacing manual entries and the "minor stoppage" black box.
* **Predictive maintenance** with vibration, temperature and current signals feeding the early-warning model; evaluate false-alarm cost vs avoided downtime.
* **True product-level costing** (ERP) instead of margin assumptions; link energy sub-meters to lines for measured, not assumed, idle consumption.
* Tailor OEE targets by line type (UHT vs water) instead of one 85% rule of thumb.

## Interview talking points
* **Why weight OEE?** Averaging shift percentages over-weights small shifts (33 pts off in the unit-test toy example, ~1 pt off on this dataset: naive 69.8% vs weighted 68.8%); weighting by planned time and volume reflects what the plant actually produced. **Is 85% right for every line?** No - it is a generic rule of thumb; UHT with long sterilisation changeovers or a new formula with a learning curve needs its own target.
* **Frequency vs severity:** page 3 ranks reasons by count and by minutes; minor jams dominate count, long breakdowns dominate minutes - they need different countermeasures.
* **Separating shift and line effects:** line fixed-effects regression (robust errors) plus bootstrap on matched line-weeks.
* **Early-warning use and cost:** the alarm feeds the weekly maintenance meeting; false alarms cost an inspection (~0.5-1 h), a missed Filler problem cost 420 hours.
* **OEE to sustainability and cost:** page 6 converts recovered minutes into kWh, water, waste and tCO2e; page 7 converts them into units and IDR.
