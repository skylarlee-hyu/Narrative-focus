"""Supplementary Table S1 (all pairwise Welch contrasts, ordinary and full samples) and the data behind Figures 1-3.
Figures 2-3 and the simple differences in Sections 3.2-3.3 are read from these contrasts."""
import _setup  # noqa: F401
import pandas as pd

from finsent import config as C
from finsent.data import load_events, samples
from finsent.stats import contrasts, mean_paths

e = load_events()
S = samples(e)

c1 = pd.concat([contrasts(S["FULL"], "FULL"), contrasts(S["ORDINARY"], "ORDINARY")], ignore_index=True)
c1.to_csv(C.TABLES / "supp_S1_pairwise_contrasts.csv", index=False)
sel = c1[(c1.Comparison == "Leader vs Firm") & (c1.Firm == "Tech")]
print(sel.round(4).to_string(index=False))

paths = pd.concat([mean_paths(S["ORDINARY"], "ORDINARY"), mean_paths(S["FULL"], "FULL")], ignore_index=True)
paths.to_csv(C.TABLES / "figure1_mean_bhar_paths.csv", index=False)
