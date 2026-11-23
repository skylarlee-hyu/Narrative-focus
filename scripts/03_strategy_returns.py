"""Section 3.4 and Appendix Table C1: simulated returns to a long-positive / short-negative strategy,
Pi(i,t,h) = D(i,t) * R(i,t,h) (i = firm, t = news day, h = horizon), by framing and firm type, market-adjusted (BHAR) and raw (ret)."""
import _setup  # noqa: F401
import pandas as pd

from finsent import config as C
from finsent.data import load_events, samples
from finsent.stats import strategy

e = load_events()
S = samples(e)
c2 = pd.concat([strategy(S["FULL"], "FULL"), strategy(S["ORDINARY"], "ORDINARY")], ignore_index=True)
c2.to_csv(C.TABLES / "appendix_C1_strategy_returns.csv", index=False)
print(c2[c2.Horizon == "t+30"][["Sample", "Framing", "Firm", "Excess", "p_excess", "Raw", "p_raw", "N"]].round(4).to_string(index=False))

# Long/short legs (Section 3.4): mean BHAR by tone within each framing x firm-type cell, ordinary sample,
# and each leg's contribution to the cell's mean strategy return (contributions sum to Pi).
o = S["ORDINARY"]
legs = []
for tech, tname in [(1, "Tech"), (0, "Non-Tech")]:
    for lead, fname in [(1, "Leader"), (0, "Firm")]:
        cell = o[(o.tech == tech) & (o.leader == lead)]
        for pos, leg in [(1, "long (positive news)"), (0, "short (negative news)")]:
            for h in C.HORIZONS:
                x = cell.loc[cell.pos == pos, f"bhar_t{h}"]
                legs.append(dict(Framing=fname, Firm=tname, Leg=leg, Horizon=f"t+{h}", N=len(x),
                                 MeanBHAR=x.mean(), ContributionToPi=(1 if pos else -1) * x.sum() / len(cell)))
legs = pd.DataFrame(legs)
legs.to_csv(C.TABLES / "strategy_legs.csv", index=False)
print(legs[legs.Horizon == "t+30"].round(4).to_string(index=False))
