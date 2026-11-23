"""Regression models of Sections 3.3 and 3.6: Eq. (1), Eq. (2), main-topic FE variant, wild cluster bootstrap,
Gelbach (2016) decomposition, Oster (2019) delta."""
import numpy as np
import pandas as pd
import pyfixest as pf

from . import config as C
from .topics import TOPIC_NAMES

TOPIC_COLS = [f"sh_{t}" for t in TOPIC_NAMES if t != "other"]  # 'other' is the omitted share


def fit(formula, data, vcov=None):
    return pf.feols(formula, data=data, vcov=vcov or C.CLUSTER)


def formulas(y, x="leader"):
    """noFE: Eq. (1) without fixed effects; eq1: Eq. (1); eq2: Eq. (2); topicFE: Eq. (1) + main-topic FE."""
    return {
        "noFE": f"{y} ~ {x} + {C.CONTROLS}",
        "eq1": f"{y} ~ {x} + {C.CONTROLS} | {C.FE}",
        "eq2": f"{y} ~ {x} + {C.CONTROLS} + {' + '.join(TOPIC_COLS)} | {C.FE}",
        "topicFE": f"{y} ~ {x} + {C.CONTROLS} | {C.FE} + main_topic",
    }


def wild_p(formula, data, x="leader", reps=C.WILD_REPS):
    """Wild cluster bootstrap p-value, firm clusters, Webb weights (Cameron, Gelbach & Miller 2008; Webb 2023)."""
    m = pf.feols(formula, data=data, vcov={"CRV1": "firm_id"})
    wb = m.wildboottest(param=x, reps=reps, cluster="firm_id", seed=C.SEED, weights_type="webb")
    return float(wb["Pr(>|t|)"])


def gelbach_boot(data, y, x="leader", reps=C.GELBACH_REPS):
    """Firm-cluster pairs bootstrap of beta(Eq. 1) - beta(Eq. 2), the Gelbach (2016) total topic contribution."""
    f = formulas(y, x)
    rng = np.random.default_rng(C.SEED)
    firms = data.company.unique()
    diffs = []
    for _ in range(reps):
        draw = rng.choice(firms, size=len(firms), replace=True)
        bs = pd.concat([data[data.company == g].assign(company=f"{g}_{i}") for i, g in enumerate(draw)])
        try:
            diffs.append(pf.feols(f["eq1"], data=bs).coef()[x] - pf.feols(f["eq2"], data=bs).coef()[x])
        except Exception:  # degenerate draw (e.g., no leader-dominant day)
            continue
    return np.array(diffs)


def gelbach_contributions(data, y, x="leader"):
    """beta1 - beta2 = sum_k theta_k * Gamma_k, k = topic shares (point estimates)."""
    f = formulas(y, x)
    m2 = pf.feols(f["eq2"], data=data)
    out = {}
    for k in TOPIC_COLS:
        g = pf.feols(f"{k} ~ {x} + {C.CONTROLS} | {C.FE}", data=data).coef()[x]
        out[k] = m2.coef()[k] * g
    return out


def oster(b1, r2_1, b2, r2_2, mult=C.OSTER_RMAX_MULT):
    """Oster (2019) delta for beta = 0 and bias-adjusted beta* at delta = 1, using within R-squared:
    Eq. (1) is the restricted model, Eq. (2) adds the topic shares as observed content controls."""
    rmax = min(1.0, mult * r2_2)
    if np.isclose(b1, b2) or np.isclose(r2_2, r2_1):
        return np.inf, b2
    delta = b2 * (r2_2 - r2_1) / ((b1 - b2) * (rmax - r2_2))
    beta_star = b2 - (b1 - b2) * (rmax - r2_2) / (r2_2 - r2_1)
    return delta, beta_star
