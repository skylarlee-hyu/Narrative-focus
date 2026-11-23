"""Appendix Table C3: topic composition of technology-firm news by narrative focus (article shares, 2021-2025).
The firm-day topic shares by narrative focus and valence (columns 4-7 of the table) are written by 13_content_controls.py."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from finsent import config as C
from finsent.data import load_events
from finsent.topics import TOPIC_NAMES


def smd(p1, p0):
    return (p1 - p0) / np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)


a = pd.read_parquet(C.ARTICLE_TOPICS_FILE)
a = a[(a.tech == 1) & (a.event_day >= C.SAMPLE_START) & (a.event_day <= C.SAMPLE_END)]
L = a[a.firm_or_leader == "Leader"].topic.value_counts(normalize=True)
F = a[a.firm_or_leader == "Company"].topic.value_counts(normalize=True)

e = load_events().merge(pd.read_parquet(C.TOPICS_FILE), on=["company", "event_day"], how="left", validate="1:1")
t5 = e[(e.tech == 1) & e.pos & (e.one == 1)]
fd = t5.groupby("leader")[[f"sh_{t}" for t in TOPIC_NAMES]].mean()

rows = [dict(topic=t, leader_articles=100 * L.get(t, 0), firm_articles=100 * F.get(t, 0),
             smd=smd(L.get(t, 0), F.get(t, 0)), leader_dominant_days=100 * fd.loc[1, f"sh_{t}"],
             firm_dominant_days=100 * fd.loc[0, f"sh_{t}"]) for t in TOPIC_NAMES]
c3 = pd.DataFrame(rows).sort_values("smd", key=lambda s: -s.abs())
c3.round(4).to_csv(C.TABLES / "appendix_C3_topic_composition.csv", index=False)
print(c3.round(2).to_string(index=False))
