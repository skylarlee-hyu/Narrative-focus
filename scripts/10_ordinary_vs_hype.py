"""Footnote 5: is the leader-firm gap different on hype days? (technology firms, positive and negative news)
The ordinary sample is 95% of the full sample, so ordinary vs. full is not a test. The clean comparison is
ordinary (non-hype) days vs. hype days, which are disjoint:
  (a) difference of the two Welch contrasts, z-test with independent-sample variances;
  (b) Leader x Hype interaction in Eq. (1) with firm and date fixed effects, two-way clustered errors."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from finsent import config as C
from finsent.data import load_events
from finsent.models import fit

e = load_events()


def gap(d, y):
    a, b = d.loc[d.leader == 1, y].dropna(), d.loc[d.leader == 0, y].dropna()
    return a.mean() - b.mean(), a.var() / len(a) + b.var() / len(b), len(a), len(b)


rows = []
for valence, pos in [("Positive", True), ("Negative", False)]:
    tp = e[(e.tech == 1) & (e.pos == pos)].copy()
    tp["leader_hype"] = tp.leader * tp.hype5
    for h in C.HORIZONS:
        y = f"bhar_t{h}"
        go, vo, _, _ = gap(tp[tp.hype5 == 0], y)
        gh, vh, n1h, n0h = gap(tp[tp.hype5 == 1], y)
        gf, _, _, _ = gap(tp, y)
        z = (go - gh) / np.sqrt(vo + vh)
        m = fit(f"{y} ~ leader + hype5 + leader_hype + {C.CONTROLS} | {C.FE}", tp)
        rows.append(dict(valence=valence, horizon=h, gap_ordinary=go, gap_full=gf, gap_hype=gh, n_leader_hype=n1h,
                         n_firm_hype=n0h, diff_ordinary_minus_hype=go - gh, z=z, p=2 * stats.norm.sf(abs(z)),
                         leader_x_hype=m.coef()["leader_hype"], leader_x_hype_p=m.pvalue()["leader_hype"]))
out = pd.DataFrame(rows)
out.to_csv(C.TABLES / "ordinary_vs_hype_days.csv", index=False)
print(out.round(4).to_string(index=False))
