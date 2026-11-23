"""Section 3 results by valence (technology firms = BBIG + semiconductors/electronics/telecom; ordinary sample).

(a) Table 2: leader-firm contrast by valence: Welch difference in means and Eq. (1) with firm and date fixed effects,
    two-way clustered standard errors, wild cluster bootstrap p (firm clusters); pooled Eq. (1) adds a positive-news
    indicator. Same estimates for non-technology firms (reported in the text).
(b) Robustness by valence (Eq. 1, t+10 and t+30): hype cutoffs, three-class argmax valence, leave-one-industry-out,
    Leader x coverage intensity (full sample).
(c) Dose-response: BHAR on continuous leader article share within firm type x valence cells (ordinary sample, valence
    rule of Section 2.2), OLS with HC1 errors and Eq. (1) with fixed effects; Leader share x Tech interaction.
(d) Appendix Table C5: the leader-firm contrast by valence under BHAR, CER (summed daily excess returns), and BHAR
    against the sample firms' average-price index instead of KOSPI.
"""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from scipy import stats

from finsent import config as C
from finsent.data import load_events

INDUSTRY = {
    "Batteries": {"LG에너지솔루션", "삼성SDI", "금양", "포스코퓨처엠", "LG화학", "SK이노베이션"},
    "Biotechnology": {"삼성바이오로직스", "셀트리온"},
    "Internet": {"네이버", "카카오"},
    "Games": {"엔씨소프트", "넷마블", "크래프톤"},
    "Semiconductors/electronics": {"삼성전자", "SK하이닉스", "LG전자"},
    "Telecommunications": {"SK텔레콤", "KT"},
}
assert set().union(*INDUSTRY.values()) == C.TECH_FIRMS

e = load_events()
e["posi"] = e.pos.astype(int)
e["has_l"] = (e.leader_article_count > 0).astype(int)   # leader tone is 0 when there is no leader-centric article
e["has_f"] = (e.firm_article_count > 0).astype(int)
o = e[e.one == 1].copy()


def eq1(y, extra=""):
    return f"{y} ~ leader + {C.CONTROLS}{extra} | {C.FE}"


def fe(f, d):
    return pf.feols(f, data=d, vcov=C.CLUSTER)


def wild(f, d, x="leader"):
    d = d.copy(); d["firm_id"] = pd.factorize(d.company)[0]
    m = pf.feols(f, data=d, vcov={"CRV1": "firm_id"})
    return float(m.wildboottest(param=x, reps=C.WILD_REPS, cluster="firm_id", seed=C.SEED,
                                weights_type="webb")["Pr(>|t|)"])


def cells(df):
    return {"Positive": (df[df.pos], ""), "Negative": (df[~df.pos], ""), "Pooled": (df, " + posi")}


# ---------------------------------------------------------------- (a) Table 2
rows = []
for ft, d0 in [("Tech", o[o.tech == 1]), ("Non-Tech", o[o.tech == 0])]:
    for val, (d, extra) in cells(d0).items():
        for h in C.HORIZONS:
            y = f"bhar_t{h}"
            rec = dict(firm_type=ft, valence=val, horizon=h, n_leader=int(d.leader.sum()), n_firm=int((d.leader == 0).sum()))
            if val != "Pooled":
                a, b = d.loc[d.leader == 1, y].dropna(), d.loc[d.leader == 0, y].dropna()
                w = stats.ttest_ind(a, b, equal_var=False)
                rec.update(welch=a.mean() - b.mean(), welch_se=(a.mean() - b.mean()) / w.statistic, welch_p=w.pvalue)
            m = fe(eq1(y, extra), d)
            rec.update(fe=m.coef()["leader"], fe_se=m.se()["leader"], fe_p=m.pvalue()["leader"], fe_n=m._N,
                       fe_wild_p=wild(eq1(y, extra), d) if ft == "Tech" else np.nan)
            rows.append(rec)
t2 = pd.DataFrame(rows)
t2.to_csv(C.TABLES / "table2_by_valence.csv", index=False)

# ---------------------------------------------------------------- (b) robustness by valence
rob = []
t_all = e[e.tech == 1]
t_ord = o[o.tech == 1]


def add(check, d_all, note="", ctrl=""):
    for val, (d, extra) in cells(d_all).items():
        for h in (10, 30):
            y = f"bhar_t{h}"
            m = fe(eq1(y, extra + ctrl), d)
            rec = dict(check=check, valence=val, horizon=h, n=m._N, n_leader=int(d.leader.sum()),
                       beta=m.coef()["leader"], se=m.se()["leader"], p=m.pvalue()["leader"], note=note)
            if val != "Pooled":
                a, b = d.loc[d.leader == 1, y].dropna(), d.loc[d.leader == 0, y].dropna()
                w = stats.ttest_ind(a, b, equal_var=False)
                rec.update(welch=a.mean() - b.mean(), welch_p=w.pvalue)
            rob.append(rec)


add("Main: ordinary sample (top 5% hype days removed)", t_ord)
add("Hype days: top 10% removed", t_all[t_all.hype10 == 0])
add("Hype days: top 20% removed", t_all[t_all.hype20 == 0])
add("Hype days: coverage >= 1.2 x 60-day mean removed", t_all[t_all.high_news_20pct == 0])
add("Full sample", t_all)
add("Valence: three-class argmax, neutral days dropped", t_ord[t_ord.argmax_neutral == 0])
add("Tone controls: indicators for leader- and firm-centric articles added", t_ord, ctrl=" + has_l + has_f")
for ind, firms in INDUSTRY.items():
    add(f"Excluding {ind}", t_ord[~t_ord.company.isin(firms)])
rb = pd.DataFrame(rob)
rb.to_csv(C.TABLES / "appendix_C4_robustness_by_valence.csv", index=False)

# leader-centered day counts by industry
cnt = t_ord.assign(industry=t_ord.company.map({f: k for k, v in INDUSTRY.items() for f in v})) \
    .groupby(["industry", "tone"]).leader.agg(["sum", "count"]).unstack()
cnt.to_csv(C.TABLES / "leader_days_by_industry.csv")

# coverage intensity (full sample), by valence
inten = []
t_all = t_all.copy(); t_all["int_c"] = t_all.log_n - t_all.log_n.mean()
for val, (d, extra) in cells(t_all).items():
    for h in (10, 30):
        m = fe(f"bhar_t{h} ~ leader + leader:int_c + {C.CONTROLS}{extra} | {C.FE}", d)
        inten.append(dict(valence=val, horizon=h, leader=m.coef()["leader"], leader_x_intensity=m.coef()["leader:int_c"],
                          se=m.se()["leader:int_c"], p=m.pvalue()["leader:int_c"]))
pd.DataFrame(inten).to_csv(C.TABLES / "robustness_intensity_by_valence.csv", index=False)

# ---------------------------------------------------------------- (c) dose-response (rule-A valence, ordinary sample)
o["bin"] = pd.cut(o.leader_article_share, [-np.inf, 0, .2, .4, .6, np.inf], labels=["0", ">0-20%", "20-40%", "40-60%", ">60%"])
bins, slopes = [], []
for (ft, tone), g in o.groupby(["ftype", "tone"]):
    for h in C.HORIZONS:
        y = f"bhar_t{h}"
        b = g.groupby("bin", observed=False)[y].agg(["mean", "sem", "count"]).reset_index()
        bins.append(b.assign(ftype=ft, tone=tone, horizon=h))
        m = smf.ols(f"{y} ~ leader_article_share", data=g).fit(cov_type="HC1")
        mf = fe(f"{y} ~ leader_article_share + {C.CONTROLS} | {C.FE}", g)
        slopes.append(dict(ftype=ft, tone=tone, horizon=f"t+{h}", slope=m.params.iloc[1], se=m.bse.iloc[1],
                           p=m.pvalues.iloc[1], n=int(m.nobs), slope_fe=mf.coef()["leader_article_share"],
                           se_fe=mf.se()["leader_article_share"], p_fe=mf.pvalue()["leader_article_share"]))
pd.concat(bins).to_csv(C.TABLES / "figure5_dose_response_bins.csv", index=False)
sl = pd.DataFrame(slopes)
sl.to_csv(C.TABLES / "appendix_C2_dose_response_slopes.csv", index=False)
inter = []
o["share_tech"] = o.leader_article_share * o.tech
for val, (d, extra) in cells(o).items():
    for h in (10, 30):
        y = f"bhar_t{h}"
        m = fe(f"{y} ~ leader_article_share + share_tech + {C.CONTROLS}{extra} | {C.FE}", d)
        mt = fe(f"{y} ~ leader_article_share + {C.CONTROLS}{extra} | {C.FE}", d[d.tech == 1])
        inter.append(dict(valence=val, horizon=h, slope_tech_fe=mt.coef()["leader_article_share"],
                          p_tech_fe=mt.pvalue()["leader_article_share"], share_x_tech=m.coef()["share_tech"],
                          p_share_x_tech=m.pvalue()["share_tech"]))
pd.DataFrame(inter).to_csv(C.TABLES / "dose_response_tests.csv", index=False)

# ---------------------------------------------------------------- (d) BHAR vs CER by valence
if C.FIRMDAY_FILE.exists():
    panel = pd.read_csv(C.FIRMDAY_FILE, usecols=["company", "event_day", "price_t0", "m_price_t0"])
    panel = panel.sort_values(["company", "event_day"])
    panel["er"] = panel.groupby("company").price_t0.pct_change() - panel.groupby("company").m_price_t0.pct_change()
    for h in C.HORIZONS:
        panel[f"cer_t{h}"] = panel.groupby("company").er.transform(
            lambda s: s[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))
    oc = o.merge(panel.drop(columns=["price_t0", "m_price_t0", "er"]), on=["company", "event_day"], how="left")
    # robustness of the benchmark: BHAR against an index formed from the average closing price of the 48 sample firms
    # on each trading day, instead of KOSPI
    px = pd.read_csv(C.FIRMDAY_FILE, usecols=["company", "event_day", "t0", "price_t0"]
                     + [f"t{h}" for h in C.HORIZONS] + [f"ret_t{h}" for h in C.HORIZONS])
    idx = px.dropna(subset=["price_t0"]).groupby("t0").price_t0.mean()
    for h in C.HORIZONS:
        px[f"bhar_sample_t{h}"] = px[f"ret_t{h}"] - (px[f"t{h}"].map(idx) / px.t0.map(idx) - 1)
    oc = oc.merge(px[["company", "event_day"] + [f"bhar_sample_t{h}" for h in C.HORIZONS]],
                  on=["company", "event_day"], how="left")
    c5 = []
    for val in ["Positive", "Negative"]:
        s = oc[(oc.tech == 1) & (oc.tone == val)]
        for h in C.HORIZONS:
            rec = dict(valence=val, horizon=f"t+{h}")
            for name, y in [("BHAR", f"bhar_t{h}"), ("CER", f"cer_t{h}"), ("BHAR_sample_index", f"bhar_sample_t{h}")]:
                a, b = s.loc[s.leader == 1, y].dropna(), s.loc[s.leader == 0, y].dropna()
                w = stats.ttest_ind(a, b, equal_var=False)
                rec.update({f"{name}_diff": a.mean() - b.mean(), f"{name}_t": w.statistic, f"{name}_p": w.pvalue})
            rec["corr"] = s[[f"bhar_t{h}", f"cer_t{h}"]].corr().iloc[0, 1]
            c5.append(rec)
    pd.DataFrame(c5).to_csv(C.TABLES / "appendix_C5_bhar_vs_cer.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t2.round(4).to_string(index=False))
print(rb.round(4).to_string(index=False))
print(cnt)
print(pd.DataFrame(inten).round(4).to_string(index=False))
print(sl.round(4).to_string(index=False))
print(pd.DataFrame(inter).round(4).to_string(index=False))
if C.FIRMDAY_FILE.exists():
    print(pd.DataFrame(c5).round(4).to_string(index=False))
