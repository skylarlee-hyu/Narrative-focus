"""Robustness of the valence rule (Section 3.6). The main rule classifies a day as positive when the dominant
article group's mean KR-FinBERT P(positive) >= P(negative), i.e. by the sign of its net tone. Here valence is
instead the three-class argmax (positive / neutral / negative), and days whose dominant class is neutral are
dropped. Leader-vs-firm Welch contrasts, ordinary sample, by firm type and valence."""
import _setup  # noqa: F401
import pandas as pd

from finsent import config as C
from finsent.data import load_events
from finsent.stats import welch

e = load_events()
o = e[(e.one == 1) & (e.argmax_neutral == 0)]
rows = []
for ft, d in o.groupby("ftype"):
    for tone, g in d.groupby("tone"):
        for h in C.HORIZONS:
            y = f"bhar_t{h}"
            r = welch(g.loc[g.leader == 1, y], g.loc[g.leader == 0, y])
            rows.append(dict(Firm=ft, Valence=tone, Horizon=f"t+{h}", **r))
out = pd.DataFrame(rows)
out.to_csv(C.TABLES / "robustness_valence_argmax.csv", index=False)
print(f"Days kept (argmax not neutral): {len(o):,} of {int((e.one == 1).sum()):,} ordinary-sample firm-days")
print(out.round(4).to_string(index=False))
