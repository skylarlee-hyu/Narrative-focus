"""Paths and fixed parameters shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("FINSENT_DATA", ROOT / "data" / "raw"))
DATA_DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

RAW_FIRMDAY_FILE = DATA_RAW / "firm_day_level_with_AR_with_ONE.csv"   # firm-day panel as delivered (49 firms)
ARTICLE_FILE = DATA_RAW / "final_article_level_dataset.csv"           # article-level file as delivered
KOSPI_FILE = DATA_RAW / "kospi_daily.csv"                             # KOSPI composite index, daily closes
FIRMDAY_FILE = DATA_DERIVED / "firm_day_kospi.csv"   # 48 KOSPI firms, returns benchmarked on KOSPI (00a_kospi_benchmark.py)
EVENTS_FILE = DATA_DERIVED / "events.parquet"          # firm-day news events with flags (built by 00_prepare.py)
TOPICS_FILE = DATA_DERIVED / "firmday_topics.parquet"  # firm-day topic shares (built by 00_prepare.py)
ARTICLE_TOPICS_FILE = DATA_DERIVED / "article_topics.parquet"

SAMPLE_START, SAMPLE_END = "2021-01-01", "2025-12-31"
MARKET_SUFFIX = ".KS"    # KOSPI-listed firms only; firms listed on another market (ticker suffix .KQ) are outside the sample
HORIZONS = [1, 3, 5, 10, 30]

# Ordinary news environment (ONE): drop each firm's top 5% of abnormal coverage,
# abnormal coverage = log(1 + articles_t) - log(1 + median articles over the prior 40 trading days).
HYPE_CUTOFF = 0.05
HYPE_ALT_CUTOFFS = {"hype10": 0.10, "hype20": 0.20}
COVERAGE_WINDOW = 40
COVERAGE_MIN_PERIODS = 20

# Controls X in Eq. (1) and Eq. (2) (Sections 3.3 and 3.6)
CONTROLS = "leader_sentiment_mean + firm_sentiment_mean + log_n + log_abn + abn_miss + multi_company_article_share"
FE = "company + event_day"
CLUSTER = {"CRV1": "company+event_day"}

SEED = 20261114
WILD_REPS = 9999
GELBACH_REPS = 199
OSTER_RMAX_MULT = 1.3

for d in (DATA_DERIVED, TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# Technology firms: the BBIG industries (batteries, biotechnology, internet, games) designated as new-growth industries
# under Korea's 2020 New Deal and tracked by the KRX K-New Deal indices, plus semiconductors, electronics, and
# telecommunications. This overrides the `tech` / `company_type` columns of the raw files (which classify LG Chem and
# SK Innovation as non-technology).
TECH_FIRMS = {
    "삼성전자", "SK하이닉스", "LG전자",                                   # semiconductors and electronics
    "SK텔레콤", "KT",                                                    # telecommunications
    "네이버", "카카오",                                                  # internet
    "엔씨소프트", "넷마블", "크래프톤",                                   # games
    "삼성바이오로직스", "셀트리온",                                      # biotechnology
    "LG에너지솔루션", "삼성SDI", "금양", "포스코퓨처엠", "LG화학", "SK이노베이션",  # batteries
}
