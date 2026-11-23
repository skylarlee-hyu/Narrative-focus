"""Compare reproduced values with those reported in the manuscript; exit with an error on any mismatch."""
import _setup  # noqa: F401
import sys

import pandas as pd

from finsent import config as C

T = C.TABLES
fails = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: reproduced {got:.4f}, manuscript {want:.4f}")
    if not ok:
        fails.append(label)


a = dict(pd.read_csv(T / "table1_panelA.csv").values)
for k, want in [("Firm-day news events", 51076), ("Total news articles", 603650)]:
    check(f"Table 1: {k}", float(str(a[k]).split()[0].replace(",", "")), want, 0)

t2 = pd.read_csv(T / "table2_by_valence.csv")
def t2v(val, h, col):
    return t2[(t2.firm_type == "Tech") & (t2.valence == val) & (t2.horizon == h)].iloc[0][col]
for val, h, col, want, tol in [("Positive", 30, "welch", -1.80, .005), ("Negative", 30, "welch", -1.43, .005),
                               ("Positive", 30, "fe", -0.86, .005), ("Negative", 30, "fe", -2.98, .005),
                               ("Pooled", 10, "fe", -0.79, .005), ("Negative", 30, "fe_wild_p", .068, .01),
                               ("Pooled", 10, "fe_wild_p", .084, .01)]:
    g = t2v(val, h, col)
    check(f"Table 2: {val} t+{h} {col}", g * 100 if col in ("welch", "fe") else g, want, tol)

cc = pd.read_csv(T / "content_controls_robustness.csv")
g = cc[(cc.valence == "Negative") & (cc.horizon == 30) & (cc.column == "(3) Eq.2")].iloc[0].beta
check("Table 3: Eq. (2), negative news, t+30", g * 100, -2.60, .005)

c1 = pd.read_csv(T / "appendix_C1_strategy_returns.csv")
for fr, want in [("Leader", -1.04), ("Firm", -0.55)]:
    g = c1[(c1.Sample == "ORDINARY") & (c1.Framing == fr) & (c1.Firm == "Tech") & (c1.Horizon == "t+30")].iloc[0].Excess
    check(f"Appendix C1: {fr}-centered technology strategy, t+30", g * 100, want, .005)

s1 = pd.read_csv(T / "storyline_1_mix_equalized_strategy.csv")
g = s1[(s1.firm_type == "Tech") & (s1.horizon == 30)].iloc[0].shortfall_mix_equalized
check("Section 3.4: mix-equalized strategy shortfall, t+30", g * 100, -1.05, .005)

c2 = pd.read_csv(T / "appendix_C2_dose_response_slopes.csv")
g = c2[(c2.ftype == "Tech") & (c2.tone == "Positive") & (c2.horizon == "t+30")].iloc[0].slope
check("Appendix C2: dose-response slope, technology, positive, t+30", g * 100, -1.80, .005)

c5 = pd.read_csv(T / "appendix_C5_bhar_vs_cer.csv")
r = c5[(c5.valence == "Positive") & (c5.horizon == "t+30")].iloc[0]
check("Appendix C5: CER, positive, t+30", r.CER_diff * 100, -1.85, .005)
check("Appendix C5: sample price index, positive, t+30", r.BHAR_sample_index_diff * 100, -1.52, .005)

if fails:
    print(f"\n{len(fails)} mismatch(es)"); sys.exit(1)
print("\nall checks passed")
