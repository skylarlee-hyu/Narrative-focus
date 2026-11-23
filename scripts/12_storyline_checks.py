"""Checks behind the main conclusion (ordinary sample, BBIG technology classification):
(1) strategy shortfall with the positive/negative mix of leader-centered days set equal to that of firm-centered days;
(2) whether the leader-firm gap differs between positive and negative news (technology firms);
(3) whether the leader-firm gap differs between technology and non-technology firms (by valence and pooled).
Welch-type tests treat firm-days as independent; FE tests use firm and date fixed effects with two-way clustering."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from finsent import config as C
from finsent.data import load_events

o = load_events()
o = o[o.one == 1].copy()
o["posi"] = o.pos.astype(int)
o["l_pos"] = o.leader * o.posi
o["l_tech"] = o.leader * o.tech
H = (5, 10, 30)


def gap(g, y):
    a, b = g.loc[g.leader == 1, y], g.loc[g.leader == 0, y]
    return a.mean() - b.mean(), a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)


def ztest(d, v):
    z = d / np.sqrt(v)
    return z, 2 * stats.norm.sf(abs(z))


def fe(formula, d):
    return pf.feols(formula, data=d, vcov={"CRV1": "company+event_day"})


rows1, rows2, rows3 = [], [], []
for h in H:
    y = f"bhar_t{h}"
    # (1) mix-equalized strategy shortfall: w_F * gap_pos - (1 - w_F) * gap_neg
    for ft, d in [("Tech", o[o.tech == 1]), ("Non-Tech", o[o.tech == 0])]:
        wF = d.loc[d.leader == 0, "posi"].mean()
        wL = d.loc[d.leader == 1, "posi"].mean()
        gp, vp = gap(d[d.pos], y)
        gn, vn = gap(d[~d.pos], y)
        adj = wF * gp - (1 - wF) * gn
        z, p = ztest(adj, wF ** 2 * vp + (1 - wF) ** 2 * vn)
        sign = np.where(d.pos, 1, -1) * d[y]
        raw_d = sign[d.leader == 1].mean() - sign[d.leader == 0].mean()
        raw_p = stats.ttest_ind(sign[d.leader == 1], sign[d.leader == 0], equal_var=False, nan_policy="omit").pvalue
        rows1.append(dict(firm_type=ft, horizon=h, pos_share_leader=wL, pos_share_firm=wF, shortfall_raw=raw_d,
                          p_raw=raw_p, shortfall_mix_equalized=adj, p_mix_equalized=p))
    # (2) positive vs negative within technology firms
    t = o[o.tech == 1]
    gp, vp = gap(t[t.pos], y)
    gn, vn = gap(t[~t.pos], y)
    z, p = ztest(gp - gn, vp + vn)
    m = fe(f"{y} ~ leader + l_pos + posi + {C.CONTROLS} | {C.FE}", t)
    rows2.append(dict(horizon=h, gap_pos=gp, gap_neg=gn, diff=gp - gn, p_welch=p,
                      fe_leader_neg=m.coef()["leader"], fe_leader_neg_p=m.pvalue()["leader"],
                      fe_extra_pos=m.coef()["l_pos"], fe_extra_pos_p=m.pvalue()["l_pos"]))
    # (3) technology vs non-technology
    for val, d in [("Positive", o[o.pos]), ("Negative", o[~o.pos]), ("Pooled", o)]:
        gt, vt = gap(d[d.tech == 1], y) if val != "Pooled" else (None, None)
        gn_, vn_ = gap(d[d.tech == 0], y) if val != "Pooled" else (None, None)
        if val == "Pooled":
            f = f"{y} ~ leader + l_tech + posi + {C.CONTROLS} | {C.FE}"
            m = fe(f, d)
            rows3.append(dict(valence=val, horizon=h, gap_tech=np.nan, gap_nontech=np.nan, diff=np.nan, p_welch=np.nan,
                              fe_leader_nontech=m.coef()["leader"], fe_leader_nontech_p=m.pvalue()["leader"],
                              fe_leader_x_tech=m.coef()["l_tech"], fe_leader_x_tech_p=m.pvalue()["l_tech"]))
        else:
            z, p = ztest(gt - gn_, vt + vn_)
            m = fe(f"{y} ~ leader + l_tech + {C.CONTROLS} | {C.FE}", d)
            rows3.append(dict(valence=val, horizon=h, gap_tech=gt, gap_nontech=gn_, diff=gt - gn_, p_welch=p,
                              fe_leader_nontech=m.coef()["leader"], fe_leader_nontech_p=m.pvalue()["leader"],
                              fe_leader_x_tech=m.coef()["l_tech"], fe_leader_x_tech_p=m.pvalue()["l_tech"]))

for name, rows in [("storyline_1_mix_equalized_strategy", rows1), ("storyline_2_pos_vs_neg_tech", rows2),
                   ("storyline_3_tech_vs_nontech", rows3)]:
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / f"{name}.csv", index=False)
    print(f"== {name}\n{df.round(4).to_string(index=False)}\n")
