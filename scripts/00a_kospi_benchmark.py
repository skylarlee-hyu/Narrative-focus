"""Benchmark returns on the KOSPI composite index and restrict the panel to KOSPI-listed firms.

Inputs (data/raw/, or the folder in FINSENT_DATA):
  firm_day_level_with_AR_with_ONE.csv  firm-day panel as delivered; its market columns (m_price_t*, m_ret_t*) hold the
                                       average closing price of the firms in the panel
  kospi_daily.csv                      KOSPI daily closes, 2021-01-04 to 2026-02-27, either as `date,close` or as one or
                                       more Investing.com exports (Korean headers 날짜, 종가) named kospi_daily*.csv
Output: data/derived/firm_day_kospi.csv, the same columns with
  m_price_t{h} = KOSPI close on t{h};  m_ret_t{h} = KOSPI(t{h}) / KOSPI(t0) - 1;
  excess_ret_t{h} = AR_t{h} = ret_t{h} - m_ret_t{h}  (buy-and-hold abnormal return, BHAR);  AR = AR_t1,
for h in {1, 2, 3, 5, 10, 15, 30}, keeping only firms whose ticker ends in C.MARKET_SUFFIX (.KS)."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from finsent import config as C

H_ALL = [1, 2, 3, 5, 10, 15, 30]


def read_kospi(folder):
    frames = []
    for f in sorted(folder.glob("kospi_daily*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        if "date" in d.columns:
            d = d[["date", "close"]]
        else:                                         # Investing.com export: 날짜 = date, 종가 = close
            d = d.rename(columns={"날짜": "date", "종가": "close"})[["date", "close"]]
        d["date"] = pd.to_datetime(d.date.str.replace(" ", ""), format="mixed").dt.strftime("%Y-%m-%d")
        d["close"] = pd.to_numeric(d.close.str.replace(",", ""))
        frames.append(d)
    k = pd.concat(frames).drop_duplicates("date").sort_values("date")
    return k.set_index("date")["close"]


kospi = read_kospi(C.DATA_RAW)
print(f"KOSPI: {len(kospi):,} trading days, {kospi.index.min()} to {kospi.index.max()}")

p = pd.read_csv(C.RAW_FIRMDAY_FILE)
dropped = sorted(p.loc[~p.ticker.str.endswith(C.MARKET_SUFFIX), "company"].unique())
p = p[p.ticker.str.endswith(C.MARKET_SUFFIX)].reset_index(drop=True)
print(f"firms kept: {p.company.nunique()}; outside KOSPI and dropped: {dropped}")

p["m_price_t0"] = p.t0.map(kospi)
for h in H_ALL:
    p[f"m_price_t{h}"] = p[f"t{h}"].map(kospi)
    p[f"m_ret_t{h}"] = p[f"m_price_t{h}"] / p.m_price_t0 - 1
    p[f"excess_ret_t{h}"] = p[f"ret_t{h}"] - p[f"m_ret_t{h}"]
    p[f"AR_t{h}"] = p[f"excess_ret_t{h}"]
p["AR"] = p.AR_t1

news = p[(p.article_count > 0) & (p.event_day >= C.SAMPLE_START) & (p.event_day <= C.SAMPLE_END)]
miss = {h: int((news[f"ret_t{h}"].notna() & news[f"excess_ret_t{h}"].isna()).sum()) for h in H_ALL}
print(f"news days: {len(news):,}; news days with a firm return but no KOSPI return, by horizon: {miss}")
assert sum(miss.values()) == 0, "KOSPI series does not cover every return window"
p.to_csv(C.FIRMDAY_FILE, index=False)
print(f"written {C.FIRMDAY_FILE} ({len(p):,} rows)")
