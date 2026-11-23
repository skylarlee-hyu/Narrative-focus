"""Illustrative forward BHAR calculation using aligned trading dates."""
import pandas as pd


def calculate_forward_bhar(
    firm_prices: pd.DataFrame,
    market_index: pd.DataFrame,
    horizons=(1, 3, 5, 10, 15, 30),
) -> pd.DataFrame:
    """Require firm_id/date/adjusted_close and date/kospi_close.

    Strictly uses market trading-day endpoints, without silently replacing a
    missing company close with a later trading date.
    """
    firms = firm_prices.copy()
    market = market_index.copy()
    firms["date"] = pd.to_datetime(firms["date"])
    market["date"] = pd.to_datetime(market["date"])
    if firms.duplicated(["firm_id", "date"]).any() or market.duplicated(["date"]).any():
        raise ValueError("Duplicate firm-day or index-day rows")

    market = market.sort_values("date").reset_index(drop=True)
    market["pos"] = range(len(market))
    firms = firms.merge(market[["date", "pos", "kospi_close"]], on="date", how="left", validate="many_to_one")
    if firms["pos"].isna().any():
        raise ValueError("Some firm dates do not match the KOSPI calendar")

    out = firms[["firm_id", "date"]].copy()
    for h in horizons:
        if int(h) != h or h < 1:
            raise ValueError("Horizons must be positive whole trading days")
        future_firm = firms[["firm_id", "pos", "adjusted_close"]].copy()
        future_firm["pos"] -= h
        future_firm = future_firm.rename(columns={"adjusted_close": "future_close"})
        future_market = market[["pos", "kospi_close"]].copy()
        future_market["pos"] -= h
        future_market = future_market.rename(columns={"kospi_close": "future_market"})
        merged = firms.merge(
            future_firm, on=["firm_id", "pos"], how="left", validate="one_to_one"
        ).merge(future_market, on="pos", how="left", validate="many_to_one")
        out[f"bhar_{h}d"] = (
            merged["future_close"] / merged["adjusted_close"] - 1
            - (merged["future_market"] / merged["kospi_close"] - 1)
        )
    return out
