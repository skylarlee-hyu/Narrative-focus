"""Section 3.6 and Table 3: do content differences explain the leader-firm gap? Technology firms (BBIG), ordinary sample,
positive news, negative news, and both pooled (pooled models add a positive-news indicator).

(1) Welch contrast (valence-specific samples only)
(2) Eq. (1): BHAR = b*Leader + g'X + firm FE + date FE
(3) Eq. (2): Eq. (1) + daily topic shares S
(4) Eq. (1) + main-topic fixed effects
(5) Eq. (2) excluding firm-days whose main topic is governance/ownership, legal/regulatory, or personnel
Standard errors clustered by firm and date; wild cluster bootstrap p (firm clusters, Webb weights) for (2)-(3).
Diagnostics for (2) -> (3): Gelbach (2016) decomposition with firm-cluster bootstrap, Oster (2019) delta.
Also: mean topic shares of leader- and firm-centered days."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from finsent import config as C
from finsent.data import load_events
from finsent.models import TOPIC_COLS, oster

LEADER_TYPICAL = {"governance_ownership", "legal_regulatory", "personnel"}
HORIZONS = (10, 30)

e = load_events().merge(pd.read_parquet(C.TOPICS_FILE), on=["company", "event_day"], how="left", validate="1:1")
t = e[(e.one == 1) & (e.tech == 1)].copy()
t["posi"] = t.pos.astype(int)
t["firm_id"] = t.company.astype("category").cat.codes
SAMPLES = {"Positive": (t[t.pos], ""), "Negative": (t[~t.pos], ""), "Pooled": (t, " + posi")}


def forms(y, extra):
    X = C.CONTROLS + extra
    return {"eq1": f"{y} ~ leader + {X} | {C.FE}",
            "eq2": f"{y} ~ leader + {X} + {' + '.join(TOPIC_COLS)} | {C.FE}",
            "topicFE": f"{y} ~ leader + {X} | {C.FE} + main_topic"}


def fe(f, d):
    return pf.feols(f, data=d, vcov=C.CLUSTER)


def wild(f, d):
    m = pf.feols(f, data=d, vcov={"CRV1": "firm_id"})
    return float(m.wildboottest(param="leader", reps=C.WILD_REPS, cluster="firm_id", seed=C.SEED,
                                weights_type="webb")["Pr(>|t|)"])


def gelbach_p(d, f):
    rng = np.random.default_rng(C.SEED)
    firms = d.company.unique()
    b1 = pf.feols(f["eq1"], data=d).coef()["leader"]
    b2 = pf.feols(f["eq2"], data=d).coef()["leader"]
    diffs = []
    for _ in range(C.GELBACH_REPS):
        draw = rng.choice(firms, size=len(firms), replace=True)
        bs = pd.concat([d[d.company == g].assign(company=f"{g}_{i}") for i, g in enumerate(draw)])
        try:
            diffs.append(pf.feols(f["eq1"], data=bs).coef()["leader"] - pf.feols(f["eq2"], data=bs).coef()["leader"])
        except Exception:
            continue
    se = np.std(diffs, ddof=1)
    return b1 - b2, 2 * stats.norm.sf(abs((b1 - b2) / se))


rows, diag = [], []
for val, (d, extra) in SAMPLES.items():
    for h in HORIZONS:
        y = f"bhar_t{h}"
        f = forms(y, extra)
        if val != "Pooled":
            a, b = d.loc[d.leader == 1, y], d.loc[d.leader == 0, y]
            w = stats.ttest_ind(a, b, equal_var=False)
            rows.append(dict(valence=val, horizon=h, column="(1) Welch", beta=a.mean() - b.mean(),
                             se=(a.mean() - b.mean()) / w.statistic, p=w.pvalue,
                             p_wild=np.nan, n=len(a) + len(b), n_leader=len(a)))
        m = {}
        for col, key in [("(2) Eq.1", "eq1"), ("(3) Eq.2", "eq2"), ("(4) Eq.1 + main-topic FE", "topicFE")]:
            m[key] = fe(f[key], d)
            rows.append(dict(valence=val, horizon=h, column=col, beta=m[key].coef()["leader"],
                             se=m[key].se()["leader"], p=m[key].pvalue()["leader"], p_wild=wild(f[key], d) if key != "topicFE" else np.nan,
                             n=m[key]._N, n_leader=int(d.leader.sum())))
        dx = d[~d.main_topic.isin(LEADER_TYPICAL)]
        m5 = fe(f["eq2"], dx)
        rows.append(dict(valence=val, horizon=h, column="(5) Eq.2 excl. governance/legal/personnel days",
                         beta=m5.coef()["leader"], se=m5.se()["leader"], p=m5.pvalue()["leader"], p_wild=wild(f["eq2"], dx), n=m5._N,
                         n_leader=int(dx.leader.sum())))
        shift, gp = gelbach_p(d, f)
        delta, bstar = oster(m["eq1"].coef()["leader"], m["eq1"]._r2_within, m["eq2"].coef()["leader"],
                             m["eq2"]._r2_within)
        diag.append(dict(valence=val, horizon=h, beta_eq1=m["eq1"].coef()["leader"],
                         beta_eq2=m["eq2"].coef()["leader"], pct_shift=shift / m["eq1"].coef()["leader"],
                         gelbach_p=gp, oster_delta=delta))

res, dg = pd.DataFrame(rows), pd.DataFrame(diag)
res.to_csv(C.TABLES / "content_controls_robustness.csv", index=False)
dg.to_csv(C.TABLES / "content_controls_diagnostics.csv", index=False)
bal = t.groupby(["tone", "leader"])[TOPIC_COLS + ["sh_other"]].mean().T
bal.columns = [f"{v}_{'leader' if l else 'firm'}" for v, l in bal.columns]
bal.to_csv(C.TABLES / "content_topic_balance_tech.csv")
pd.set_option("display.width", 200)
print(res.assign(beta=(100 * res.beta).round(2)).round(3).to_string(index=False))
print(dg.round(3).to_string(index=False))
print((100 * bal).round(1).to_string())
