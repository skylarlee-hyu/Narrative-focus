# Beyond Sentiment: Narrative Focus and Stock Returns

**Research repository — development stage**  
**Author:** Sunyoung Lee · Hanyang University

This research examines whether the actor foregrounded in financial news—a corporate leader or the firm—is associated with subsequent stock market returns beyond news sentiment.

## Study design

The current manuscript describes **more than 600,000 Korean financial-news records** relating to **48 large KOSPI-listed firms** during **2021–2025**. Final sample statistics and manuscript results should be verified against the locked analysis data.

- **Narrative focus:** Identified from **headline wording**, using references to executives' names or titles. It is **not classified by a large language model**.
- **News sentiment:** Measured separately with a Korean financial-text transformer model.
- **Outcome:** Buy-and-hold abnormal returns (BHARs), examined over 1, 3, 5, 10, 15 and 30 trading days.
- **Heterogeneity:** Positive versus negative news, and technology versus non-technology firms.
- **Further analysis:** Valence-following investment strategy and robustness checks.

## Repository contents

- `src/` — starter headline and return calculations
- `scripts/` — KOSPI index retrieval helper
- `docs/` — measurement notes and reproducibility checklist
- `data/` — local files only; raw and derived restricted data are ignored by Git

## Status

**This is a research scaffold, not a verified replication archive.** No raw news articles, licensed market-price data, unpublished manuscript, confidential materials or validated empirical results are published here.

The current benchmark reconstruction uses the daily KOSPI index rather than the arithmetic average of sample firms' share prices. Return-construction conventions (including corporate actions, dividends and news-day alignment) must be checked before rebuilding the manuscript tables.

## Running the code

Install Python 3.10+ and dependencies:

```bash
pip install pandas numpy finance-datareader
```

Download historical KOSPI levels:

```bash
python scripts/fetch_kospi.py
```

Review `docs/replication_checklist.md` before using the code for empirical claims.

## Citation

Sunyoung Lee, *Beyond Sentiment: Narrative Focus and Stock Returns*, manuscript in preparation. Update the citation when published.

## Data availability

Underlying news texts and security-level price data are excluded pending a review of licensing and redistribution rights.
