# Technical replication guide

These notes document the full analytical workflow, definitions, assumptions, and script-to-output mapping. For the project overview, see the [repository homepage](../README.md).

This repository reproduces the tables, figures, and in-text statistics of *Beyond Sentiment: Narrative Focus and
Stock Returns*: 48 KOSPI-listed firms, news days from January 2021 to December 2025, buy-and-hold abnormal returns
(BHARs) measured against the KOSPI composite index.

## Quick start

```bash
pip install -r requirements.txt
# put the three raw files in data/raw/ (see data/README.md), or set FINSENT_DATA=/path/to/folder
./run_all.sh
```

`run_all.sh` runs the scripts in order, writes tables to `outputs/tables/`, figures to `outputs/figures/`, and logs
to `outputs/logs/`, and ends with `99_verify.py`, which compares key reproduced values with those reported in the
manuscript and exits with an error on any mismatch. A full run takes about 10–15 minutes on a laptop, most of it
the wild cluster bootstrap.

## Scripts and outputs

| Script | Manuscript output | Files in `outputs/` |
|---|---|---|
| `00a_kospi_benchmark.py` | Sample of 48 KOSPI firms; BHARs against KOSPI (Sections 2.1, 3.1.1) | `data/derived/firm_day_kospi.csv` |
| `00_prepare.py` | Ordinary-sample flags (Section 3.1.2), narrative focus and valence (Section 2.2), topic labels | `data/derived/*.parquet` |
| `01_table1_descriptives.py` | Table 1, Panels A–C | `table1_panelA/B/C.csv` |
| `02_framing_contrasts.py` | Supplementary Table S1 (all 120 pairwise Welch contrasts); simple differences in Sections 3.2–3.3; data for Figures 1–3 | `supp_S1_pairwise_contrasts.csv`, `figure1_mean_bhar_paths.csv` |
| `03_strategy_returns.py` | Section 3.4; Appendix Table C1 | `appendix_C1_strategy_returns.csv`, `strategy_legs.csv` |
| `05_topic_composition.py` | Appendix Table C3, article shares | `appendix_C3_topic_composition.csv` |
| `08_heterogeneity_tests.py` | Footnotes 6–7 (positive vs. negative gap; technology vs. non-technology gap); strategy shortfall (Section 3.4) | `heterogeneity_tests.csv`, `strategy_leader_minus_firm.csv` |
| `10_ordinary_vs_hype.py` | Footnote 5 (leader–firm gap on hype days vs. ordinary days) | `ordinary_vs_hype_days.csv` |
| `11_valence_argmax.py` | Section 3.6: valence by the most probable class, differences in means | `robustness_valence_argmax.csv` |
| `12_storyline_checks.py` | Section 3.4 (news mix equalized); Sections 3.2.1–3.2.2 and footnotes 6–7 | `storyline_1/2/3_*.csv` |
| `13_content_controls.py` | Table 3 (Eq. 2, Gelbach, Oster); Appendix Table C3, firm-day topic shares | `content_controls_robustness.csv`, `content_controls_diagnostics.csv`, `content_topic_balance_tech.csv` |
| `14_results_by_valence.py` | Table 2; Section 3.5 and Appendix Table C2 (dose–response); Appendix Table C4 (robustness); Appendix Table C5 (BHAR, CER, sample price index); coverage-intensity interaction (Section 3.6) | `table2_by_valence.csv`, `appendix_C2_dose_response_slopes.csv`, `figure5_dose_response_bins.csv`, `dose_response_tests.csv`, `appendix_C4_robustness_by_valence.csv`, `robustness_intensity_by_valence.csv`, `appendix_C5_bhar_vs_cer.csv` |
| `15_figures.py` | Figures 1–5 | `figures/figure1_…png` to `figure5_…png` |
| `99_verify.py` | Checks reproduced values against the manuscript | log only |

Shared code is in `src/finsent/`: `config.py` (paths and fixed parameters), `data.py` (sample and
classification), `stats.py` (Welch tests, strategy returns), `models.py` (fixed-effects regressions, wild cluster
bootstrap, Gelbach, Oster), `topics.py` (keyword topic dictionary).

## Definitions implemented

**Sample.** The 48 KOSPI-listed firms of the panel; firm-days with at least one article between 2021-01-01 and
2025-12-31 (51,076 news days). Days on which the leader and firm article shares are equal (548) are dropped from
all contrasts and models.

**Outcome.** Buy-and-hold abnormal return, `bhar_t{h}`: the compounded firm return from the close of news day t to
day t+h minus the KOSPI return over the same days, h ∈ {1, 3, 5, 10, 30}. The news-day return itself is not used.
Appendix Table C5 repeats the main contrast with the cumulative excess return (CER, summed daily excess returns)
and with an index formed from the average closing price of the 48 sample firms instead of KOSPI.

**Narrative focus.** An article is leader-centric when its headline contains a person's name or an executive title
(headline only), firm-centric otherwise. A firm-day is leader-centered if `leader_article_share >
firm_article_share` and firm-centered otherwise; ties are dropped.

**Sentiment, tone, and valence.** KR-FinBERT-SC class probabilities. Tone is the net tone of each article group
(`leader_sentiment_mean`, `firm_sentiment_mean` = mean P(positive) − mean P(negative)) and is a control in Eq. (1)–(2).
Valence is the sign of the net tone of the dominant article group, by one rule for both groups
(`src/finsent/data.py`). `11_valence_argmax.py` and `14_results_by_valence.py` repeat the main contrasts with the
three-class argmax, dropping days whose most probable class is neutral.

**Ordinary sample.** For each firm, abnormal coverage is log(1 + articles_t) − log(1 + median articles over the
prior 40 trading days), computed on the full firm-day panel (minimum 20 prior days). A news day is a hype day when
its abnormal coverage exceeds the firm's 95th percentile across its 2021–2025 news days (2,496 days, 4.9%). The
ordinary sample (`one` = 1) contains the other 48,580 news days. Alternative cutoffs (10%, 20%, and coverage ≥ 1.2 ×
the 60-day mean) are used in Appendix Table C4. The `ONE` column in the raw firm-day file is an older, return-based
definition and is not used.

**Technology firms.** The BBIG industries (secondary batteries, biotechnology, internet platforms, games) plus
semiconductors and electronics and telecommunications: 18 firms, listed in `TECH_FIRMS` in `src/finsent/config.py`;
the other 30 firms are non-technology firms.

**Models.** Eq. (1): `BHAR(i,t,h) = β·Leader + γ′X + α_firm + δ_date + ε`; Eq. (2) adds the topic shares S (ten
topics; "other" omitted). X = leader and firm tone, log(1 + article count), log(1 + abnormal news ratio) with a
missing indicator, and the share of multi-company articles. Standard errors are clustered by firm and date; wild
cluster bootstrap p-values use firm clusters, 9,999 draws, Webb weights, seed 20261114. Gelbach (2016)
decomposition with a 199-draw firm-cluster bootstrap; Oster (2019) δ with within R² and R²max = 1.3 × R².

**Topics.** `src/finsent/topics.py` assigns each article to one of ten topics or "other" from keywords in the
headline (weight 2) and the first 120 characters of the body (weight 1). Person words such as 회장 (chairman) are
not keywords, so naming the leader cannot change an article's topic.

## Inference note

Welch tests and one-sample t-tests (Supplementary Table S1, Appendix Table C1, and the simple differences in the
text) treat firm-days as independent. Dependence-robust inference (firm and date fixed effects, two-way clustering,
wild cluster bootstrap) is reported in Tables 2–3 and Appendix Tables C2 and C4.

## Software

Python 3.13 with the package versions in `requirements.txt`. Results were produced with pandas 3.0.5, pyfixest 0.70.0,
and wildboottest 0.3.2; bootstrap p-values can differ slightly across versions. Fixed-effects estimation requires
pandas 3.x with this pyfixest version.
