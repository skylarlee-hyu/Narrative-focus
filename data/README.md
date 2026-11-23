# Data

## Raw inputs (not included; place in `data/raw/` or point `FINSENT_DATA` to their folder)

| File | Unit | Source | Used for |
|---|---|---|---|
| `firm_day_level_with_AR_with_ONE.csv` | firm × calendar day, 49 firms as delivered (63,945 rows, including zero-coverage days) | Firm-day panel built from BIGKinds news (Korea Press Foundation, retrieved December 2025) and daily closing prices | all tables and figures |
| `final_article_level_dataset.csv` | article × firm (608,532 rows as delivered) | BIGKinds; KR-FinBERT-SC class probabilities | topic labels (Eq. 2; Appendix Table C3), article counts (Table 1) |
| `kospi_daily.csv` | trading day, 2021-01-04 to 2026-02-27 (1,263 days) | KOSPI composite index daily closes, downloaded from Investing.com (kr.investing.com/indices/kospi-historical-data) | market benchmark |

`kospi_daily.csv` may be a two-column file (`date,close`) or one or more Investing.com exports saved as
`kospi_daily*.csv` (Korean headers 날짜, 종가; thousands separators are handled). The news texts are licensed
and are not redistributed.

The analysis sample is the 48 KOSPI-listed firms of the panel (ticker suffix `.KS`); `00a_kospi_benchmark.py`
drops any firm listed on another market. Variable definitions are in Appendix A of the manuscript.

## Derived files (`data/derived/`, created by the scripts)

- `firm_day_kospi.csv` (`00a_kospi_benchmark.py`): the firm-day panel for the 48 KOSPI firms, with the market
  columns replaced by KOSPI: `m_price_t{h}` = KOSPI close on day t+h, `m_ret_t{h}` = KOSPI(t+h)/KOSPI(t) − 1,
  `excess_ret_t{h}` = `AR_t{h}` = `ret_t{h}` − `m_ret_t{h}` (the BHAR), for h ∈ {1, 2, 3, 5, 10, 15, 30}.
- `events.parquet` (`00_prepare.py`): firm-day news events with the ordinary-sample flags (`abn_cov`, `hype5`,
  `hype10`, `hype20`, `one`), narrative focus (`leader`, `frame`), valence (`pos`, `tone`), firm type, and controls.
- `article_topics.parquet`: topic label for every article; `firmday_topics.parquet`: daily topic shares (`sh_*`)
  and main topic for every firm-day.
