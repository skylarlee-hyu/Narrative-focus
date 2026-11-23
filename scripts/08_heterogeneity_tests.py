"""Formal tests of differences behind the heterogeneity claims (ordinary sample):
(a) technology vs. non-technology: difference in the leader-firm gap after positive news;
(b) positive vs. negative news: difference in the leader-firm gap within technology firms.
Each is reported as a difference of Welch contrasts (independent groups) and as an interaction in a
regression with firm and date fixed effects and two-way clustered standard errors."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from finsent import config as C
from finsent.data import load_events
from finsent.models import fit

e = load_events()
o = e[e.one == 1].copy()
o["tech_leader"] = o.tech * o.leader
o["pos_leader"] = o.pos.astype(int) * o.leader
o["posi"] = o.pos.astype(int)


def gap(d, y):
    a, b = d.loc[d.leader == 1, y].dropna(), d.loc[d.leader == 0, y].dropna()
    return a.mean() - b.mean(), a.var() / len(a) + b.var() / len(b)


rows = []
for h in C.HORIZONS:
    y = f"bhar_t{h}"
    for name, g1, g2, inter, data in [
        ("tech minus non-tech (positive news)", o[(o.tech == 1) & o.pos], o[(o.tech == 0) & o.pos], "tech_leader", o[o.pos]),
        ("positive minus negative (tech firms)", o[(o.tech == 1) & o.pos], o[(o.tech == 1) & ~o.pos], "pos_leader", o[o.tech == 1]),
    ]:
        (d1, v1), (d2, v2) = gap(g1, y), gap(g2, y)
        diff, se = d1 - d2, np.sqrt(v1 + v2)
        main = "tech" if inter == "tech_leader" else "posi"
        f = f"{y} ~ leader + {inter} + {C.CONTROLS} | {C.FE}" if main == "tech" else f"{y} ~ leader + posi + {inter} + {C.CONTROLS} | {C.FE}"
        m = fit(f, data)
        rows.append(dict(test=name, horizon=h, gap1=d1, gap2=d2, diff=diff, z=diff / se, p_welch=2 * stats.norm.sf(abs(diff / se)),
                         interaction=m.coef()[inter], interaction_se=m.se()[inter], interaction_p=m.pvalue()[inter]))
out = pd.DataFrame(rows)
out.to_csv(C.TABLES / "heterogeneity_tests.csv", index=False)
print(out.round(4).to_string(index=False))

# (c) leader- vs firm-framed strategy return, by firm type (Welch test on Pi = D x BHAR)
o["Pi_sign"] = np.where(o.pos, 1.0, -1.0)
srows = []
for ft in [1, 0]:
    for h in C.HORIZONS:
        d = o[o.tech == ft]
        pi = d.Pi_sign * d[f"bhar_t{h}"]
        a, b = pi[d.leader == 1].dropna(), pi[d.leader == 0].dropna()
        t = stats.ttest_ind(a, b, equal_var=False)
        srows.append(dict(firm_type="Tech" if ft else "Non-Tech", horizon=h, leader=a.mean(), firm=b.mean(),
                          diff=a.mean() - b.mean(), t=t.statistic, p=t.pvalue))
s = pd.DataFrame(srows)
s.to_csv(C.TABLES / "strategy_leader_minus_firm.csv", index=False)
print(s.round(4).to_string(index=False))
