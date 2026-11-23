"""Build derived data (run after 00a_kospi_benchmark.py): firm-day events with ordinary-sample flags; article topic labels; firm-day topic shares."""
import _setup  # noqa: F401
import pandas as pd

from finsent import config as C
from finsent.data import build_events
from finsent.topics import TOPIC_NAMES, classify

events = build_events()
events.to_parquet(C.EVENTS_FILE, index=False)
print(f"events: {len(events):,} news days; hype days (top 5%): {events.hype5.sum():,} "
      f"({events.hype5.mean():.1%}); ties: {events.tie.sum():,}")

cols = ["news_id", "company", "tech", "firm_or_leader", "title", "content", "event_day"]
a = pd.read_csv(C.ARTICLE_FILE, usecols=cols)
a = a[a.company.isin(events.company.unique())]          # the 48 KOSPI firms of the panel
a["tech"] = a.company.isin(C.TECH_FIRMS).astype(int)   # BBIG + semiconductors/electronics/telecom
a["topic"] = classify(a.title, a.content.astype(str).str.slice(0, 300))
a.drop(columns=["content", "title"]).to_parquet(C.ARTICLE_TOPICS_FILE, index=False)

sh = pd.crosstab([a.company, a.event_day], a.topic, normalize="index").reindex(columns=TOPIC_NAMES, fill_value=0.0)
main = sh.idxmax(axis=1).where(sh.max(axis=1) >= 0.5, "mixed")       # main topic: share >= 50%, else "mixed"
sh.columns = [f"sh_{c}" for c in sh.columns]
sh["main_topic"] = main
sh.reset_index().to_parquet(C.TOPICS_FILE, index=False)
print(f"articles labeled: {len(a):,}; firm-days with topic shares: {len(sh):,}")
