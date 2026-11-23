"""Firm-day news events: sample, hype flags (ONE), framing and tone classification, controls."""
import numpy as np
import pandas as pd

from . import config as C


def abnormal_coverage(panel: pd.DataFrame) -> pd.Series:
    """log(1 + N_t) - log(1 + median N over the prior 40 trading days), by firm.
    Computed on the full firm-day panel (including zero-coverage days) so the benchmark reflects
    each firm's normal coverage."""
    p = panel.sort_values(["company", "event_day"])
    med = p.groupby("company").article_count.transform(
        lambda s: s.shift(1).rolling(C.COVERAGE_WINDOW, min_periods=C.COVERAGE_MIN_PERIODS).median())
    return (np.log1p(p.article_count) - np.log1p(med)).reindex(panel.index)


def hype_flag(df: pd.DataFrame, cutoff: float) -> pd.Series:
    """1 if a news day's abnormal coverage is in the firm's top `cutoff` share of news days."""
    thr = df.groupby("company").abn_cov.transform(lambda x: x.quantile(1 - cutoff))
    return (df.abn_cov > thr).astype(int)


def build_events(path=C.FIRMDAY_FILE) -> pd.DataFrame:
    """All firm-day news events in 2021-2025 (ties between leader and firm share kept; see `tie`)."""
    panel = pd.read_csv(path)
    panel["tech"] = panel.company.isin(C.TECH_FIRMS).astype(int)   # BBIG + semiconductors/electronics/telecom
    panel["abn_cov"] = abnormal_coverage(panel)
    s = panel[(panel.article_count > 0) & (panel.event_day >= C.SAMPLE_START)
              & (panel.event_day <= C.SAMPLE_END)].copy()
    s["hype5"] = hype_flag(s, C.HYPE_CUTOFF)
    for name, q in C.HYPE_ALT_CUTOFFS.items():
        s[name] = hype_flag(s, q)
    s["one"] = 1 - s.hype5                                          # ordinary news environment

    # framing: leader-dominant if leader share > firm share; ties are dropped from all contrasts
    s["tie"] = (s.leader_article_share == s.firm_article_share).astype(int)
    s["leader"] = (s.leader_article_share > s.firm_article_share).astype(int)
    s["frame"] = np.where(s.leader == 1, "Leader", "Firm")
    # valence: sign of the net tone of the dominant article group, applied identically to leader- and
    # firm-centered days: positive iff the group's mean KR-FinBERT P(positive) >= mean P(negative).
    # Leader-dominant days use the leader-centric articles, firm-dominant days the firm-centric articles.
    lp, ln_ = s.leader_finbert_positive, s.leader_finbert_negative
    fp, fn = s.firm_finbert_positive, s.firm_finbert_negative
    s["pos"] = np.where(s.leader == 1, lp >= ln_, fp >= fn)
    # robustness: three-class argmax; flags days whose dominant class is neutral (dropped in the check)
    lu, fu = s.leader_finbert_neutral, s.firm_finbert_neutral
    p_, n_, u_ = np.where(s.leader == 1, lp, fp), np.where(s.leader == 1, ln_, fn), np.where(s.leader == 1, lu, fu)
    s["argmax_neutral"] = ((u_ > p_) & (u_ > n_)).astype(int)
    s["tone"] = np.where(s.pos, "Positive", "Negative")
    s["ftype"] = np.where(s.tech == 1, "Tech", "Non-Tech")

    # outcome: buy-and-hold abnormal return (BHAR) = compounded firm return - compounded return on the
    # KOSPI composite index from day 1 to day h (excess_ret_t{h} = ret_t{h} - m_ret_t{h} in data/derived/firm_day_kospi.csv,
    # where m_ret_t{h} is the KOSPI return; see scripts/00a_kospi_benchmark.py)
    for h in [1, 2, 3, 5, 10, 15, 30]:
        s[f"bhar_t{h}"] = s[f"excess_ret_t{h}"]

    # controls X
    s["log_n"] = np.log1p(s.article_count)
    s["abn_miss"] = s.abnormal_news_ratio.isna().astype(int)
    s["log_abn"] = np.log1p(s.abnormal_news_ratio.fillna(0))
    s["firm_id"] = pd.factorize(s.company)[0]
    return s.reset_index(drop=True)


def load_events(drop_ties=True) -> pd.DataFrame:
    e = pd.read_parquet(C.EVENTS_FILE)
    return e[e.tie == 0].copy() if drop_ties else e


def samples(e: pd.DataFrame) -> dict:
    return {"ORDINARY": e[e.one == 1], "FULL": e}
