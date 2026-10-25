# FINDINGS - Production Efficiency (OEE) & Downtime Analytics

*Synthetic data for portfolio purposes (seed 42). Money values rest on illustrative assumptions in `src/config.py` (editable in the dashboard sidebar). All figures below are computed by `scripts/build_findings.py`.*

## Headline numbers

1. **Plant-wide OEE is 68.8%** (A = 85.1%, P = 83.0%, Q = 97.4%), weighted by summed minutes and units. The largest loss bucket is **Reduced speed** (11.1% of planned time), followed by Breakdown (9.1%). Only 69.9% of planned time is fully productive. Lines range from 63.3% (L02) to 78.3% (L04).
2. **L02's Filler** caused **420 hours** of breakdown between Apr-Sep 2025 (~70 h/month vs ~2 h/month after the fix). L02 OEE recovered from **57.0%** (Apr-Sep) to **70.1%** (Oct-Dec): **+13.1 pts** (95% bootstrap CI 10.8 to 15.2, resampling weeks).
3. **L05 UHT changeovers average 92 min** (median 91, P90 120) vs ~51 min on the PET lines. Cutting them by 30% would recover ~98 h and add ~0.59 M units/yr (~IDR 1.14 bn).
4. **Night shift (S3) OEE is 5.3 pts below the day shifts** (65.2% vs 70.5%); within the same line the regression gap vs S1 is -5.2 pts (95% CI -6.0 to -4.4, p < 0.001). Performance is 3.2 pts lower, availability 2.8 pts lower, quality 0.6 pts lower; Waiting-operator stops: 412 on S3 vs 222 (S1) and 235 (S2).
5. **Early warning:** a robust weekly threshold (median + 3 MAD of a trailing non-alarm baseline) first flagged L02 in the week of **07 Apr 2025**, **10 weeks before** its worst 4-week window (ending 16 Jun 2025). IsolationForest flagged it 10 weeks before. Other lines raised 17 alarm-weeks in total (false-alarm cost is visible on page 5).

## Other embedded patterns the dashboard recovers

- **L06 material shortage:** 153 shortage events in Jul-Sep vs 19/month in other months.
- **Start-up after idle** (482 shifts): breakdown rate 1.25x (95% CI 1.15-1.35); reject rate 3.40% vs 2.52%.
- **FRM-900 learning curve (L06):** performance starts at 71.2% in the first week and reaches 84.3% by weeks 9-12.
- **Rainy season:** utility stop minutes per operating hour are 1.63x higher in Nov-Mar than Apr-Oct.
- **Energy intensity:** 55.2 kWh per 1,000 good units plant-wide (CO2e uses an assumed 0.85 kg/kWh grid factor).

## Ranked improvement opportunities (annual, default assumptions)

| # | Action | Hours/yr | Extra good units/yr | Contribution/yr | Effort (1-5) |
|---|---|---|---|---|---|
| 1 | Night-shift staffing & skills plan | 357 | 3.16 M | IDR 5.33 bn | 3 |
| 2 | L02 Filler: sustain the overhaul + PM routine | 812 | 6.45 M | IDR 3.83 bn | 2 |
| 3 | Minor-stoppage elimination on top reasons | 167 | 1.12 M | IDR 2.20 bn | 2 |
| 4 | Supplier review for L06 material shortages | 36 | 0.11 M | IDR 1.91 bn | 2 |
| 5 | SMED on L05 UHT changeovers | 98 | 0.59 M | IDR 1.14 bn | 3 |
| 6 | Start-up-after-idle checklist & warm-up run | 38 | 0.33 M | IDR 537 M | 1 |
| 7 | Rainy-season utilities protection | 29 | 0.17 M | IDR 312 M | 3 |

Simple sum: IDR 15.26 bn per year - an **upper bound** because levers overlap (e.g. a Filler fix also lifts night-shift OEE) and every IDR figure depends on the illustrative contribution-margin table.

## Caveats

- Synthetic data: effects were embedded by a generator, so significance tests demonstrate method, not real-world truth.
- Before/after (L02) has no control line; other changes in the same period could contribute.
- p-charts: with 50-100k units per shift a classic p-chart flags most shifts; the dashboard defaults to Laney p' for that reason.
- The next-shift breakdown model is weak by design of the data (precision close to base rate) - the weekly alarm and risk-score watch-list are the useful tools.
