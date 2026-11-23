"""Convenience source for historical KOSPI levels; verify against official KRX."""
from pathlib import Path
import FinanceDataReader as fdr

START = "2020-12-01"
END = "2026-03-01"
DEST = Path("data/raw/kospi_daily.csv")

def main():
    df = fdr.DataReader("KS11", START, END)[["Close"]].rename(
        columns={"Close": "kospi_close"}
    )
    df.index.name = "date"
    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST)
    print(f"Saved {len(df)} KOSPI observations to {DEST}")

if __name__ == "__main__":
    main()
