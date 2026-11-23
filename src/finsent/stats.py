"""Welch contrasts and signal-following strategy returns."""
import numpy as np
import pandas as pd
from scipy import stats

from .config import HORIZONS


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def welch(a: pd.Series, b: pd.Series):
    """Mean difference, Welch t and p, group sizes."""
    a, b = a.dropna(), b.dropna()
    t = stats.ttest_ind(a, b, equal_var=False)
    return dict(MeanDiff=a.mean() - b.mean(), t=t.statistic, p=t.pvalue, N1=len(a), N2=len(b))


def _specs():
    """The 12 comparisons of Appendix Table C1, in table order: (comparison, row label, condition, group1, group2)."""
    out = []
    for f in ["Non-Tech", "Tech"]:
        for c in ["Negative", "Positive"]:
            out.append(("Leader vs Firm", f, c, dict(ftype=f, tone=c, frame="Leader"), dict(ftype=f, tone=c, frame="Firm")))
    for f in ["Non-Tech", "Tech"]:
        for c in ["Firm", "Leader"]:
            out.append(("Positive vs Negative", f, c, dict(ftype=f, frame=c, tone="Positive"),
                        dict(ftype=f, frame=c, tone="Negative")))
    for f in ["Firm", "Leader"]:
        for c in ["Negative", "Positive"]:
            out.append(("Tech vs Non-Tech", f, c, dict(frame=f, tone=c, ftype="Tech"), dict(frame=f, tone=c, ftype="Non-Tech")))
    return out


def _select(df, cond):
    m = np.ones(len(df), bool)
    for k, v in cond.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def contrasts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for comp, first, cond, g1, g2 in _specs():
        a, b = _select(df, g1), _select(df, g2)
        for h in HORIZONS:
            r = welch(a[f"bhar_t{h}"], b[f"bhar_t{h}"])
            rows.append(dict(Sample=label, Comparison=comp, Firm=first, Cond=cond, Horizon=f"t+{h}", **r, Sig=stars(r["p"])))
    return pd.DataFrame(rows)


def strategy(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Pi(i,t,h) = D(i,t) * R(i,t,h) (i = firm, t = news day, h = horizon), D = +1 after positive tone, -1 after negative tone; one-sample t vs 0."""
    rows = []
    D = np.where(df.pos, 1.0, -1.0)
    for fr in ["Firm", "Leader"]:
        for ft in ["Non-Tech", "Tech"]:
            m = ((df.frame == fr) & (df.ftype == ft)).to_numpy()
            for h in HORIZONS:
                ex = pd.Series(D[m] * df.loc[m, f"bhar_t{h}"].to_numpy()).dropna()
                raw = pd.Series(D[m] * df.loc[m, f"ret_t{h}"].to_numpy()).dropna()
                te, tr = stats.ttest_1samp(ex, 0), stats.ttest_1samp(raw, 0)
                rows.append(dict(Sample=label, Framing=fr, Firm=ft, Horizon=f"t+{h}",
                                 Excess=ex.mean(), t_excess=te.statistic, p_excess=te.pvalue, Sig_excess=stars(te.pvalue),
                                 Raw=raw.mean(), t_raw=tr.statistic, p_raw=tr.pvalue, Sig_raw=stars(tr.pvalue),
                                 N=int(m.sum())))
    return pd.DataFrame(rows)


def mean_paths(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.groupby(["ftype", "frame", "tone"])[[f"bhar_t{h}" for h in HORIZONS]].mean().reset_index()
    out.insert(0, "Sample", label)
    return out
