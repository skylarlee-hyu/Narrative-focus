"""Table 1: sample composition (Panel A), buy-and-hold abnormal returns, BHAR (Panel B), news and sentiment variables (Panel C).
All firm-day news events in 2021-2025, ties included (51,076 events, 48 KOSPI firms)."""
import _setup  # noqa: F401
import pandas as pd

from finsent import config as C
from finsent.data import load_events

e = load_events(drop_ties=False)
n = len(e)
A = [
    ("Unique firms (tech / non-tech)", f"{e.company.nunique()} ({e[e.tech == 1].company.nunique()} / {e[e.tech == 0].company.nunique()})"),
    ("Sample period", f"{e.event_day.min()} to {e.event_day.max()}"),
    ("Firm-day news events", f"{n:,}"),
    ("Technology / non-technology", f"{(e.tech == 1).sum():,} ({(e.tech == 1).mean():.1%}) / {(e.tech == 0).sum():,}"),
    ("Ordinary-sample events (ONE = 1)", f"{e.one.sum():,} ({e.one.mean():.1%})"),
    ("Total news articles", f"{int(e.article_count.sum()):,}"),
    ("Leader-centric / firm-centric", f"{int(e.leader_article_count.sum()):,} "
                                      f"({e.leader_article_count.sum() / e.article_count.sum():.1%}) / "
                                      f"{int(e.firm_article_count.sum()):,}"),
    ("Leader-dominant events", f"{e.leader.sum():,} ({e.leader.mean():.1%})"),
    ("Mean events per firm", f"{n / e.company.nunique():.0f}"),
]
pd.DataFrame(A, columns=["Item", "Value"]).to_csv(C.TABLES / "table1_panelA.csv", index=False)


def summ(x):
    q = x.quantile([.25, .5, .75])
    return dict(N=int(x.count()), Mean=x.mean(), SD=x.std(), p25=q[.25], Median=q[.5], p75=q[.75])


B = pd.DataFrame([dict(Horizon=f"BHAR (t+{h})", **summ(100 * e[f"bhar_t{h}"])) for h in C.HORIZONS])
B.round(3).to_csv(C.TABLES / "table1_panelB.csv", index=False)

VARS = {"article_count": "Article count (daily)", "leader_article_share": "Leader article share",
        "firm_article_share": "Firm article share", "leader_finbert_positive": "Leader sentiment: positive",
        "leader_finbert_negative": "Leader sentiment: negative", "firm_finbert_positive": "Firm sentiment: positive",
        "firm_finbert_negative": "Firm sentiment: negative", "abnormal_news_ratio": "Abnormal news ratio"}
P = pd.DataFrame([dict(Variable=lab, **summ(e[v])) for v, lab in VARS.items()]).drop(columns="N")
P.round(3).to_csv(C.TABLES / "table1_panelC.csv", index=False)
print(pd.DataFrame(A).to_string(index=False, header=False)); print(B.round(3).to_string(index=False))
