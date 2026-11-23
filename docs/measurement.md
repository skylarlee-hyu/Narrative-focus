# Measurement notes

## Narrative focus

This is a **rule-based headline measure** built from visible executive names and titles. It must not be described as LLM-based classification. An executive mention is not necessarily equivalent to a psychological framing effect. A headline without an executive reference is not automatically firm-centered; the comparison category must be justified in the final codebook.

## News sentiment

A separate transformer trained for Korean financial language produces sentiment. Record the exact checkpoint, preprocessing, treatment of neutral items and score aggregation.

## Buy-and-hold abnormal return

For a firm with prices aligned to the same endpoints as the market index, the market-adjusted BHAR over h trading days is

```text
BHAR(i,t,h) = (P(i,t+h)/P(i,t) - 1) - (M(t+h)/M(t) - 1)
```

The formula assumes appropriately adjusted firm prices. Verify whether dividends are included, and match the benchmark's return convention. Do not calculate BHAR by summing daily excess returns.

The prior benchmark based on `stock_close.mean(axis=1)` is an average of nominal share prices, not a broad-market index.

## Interpretation

The study reports conditional associations between headline narrative focus and subsequent returns. It does not directly establish psychological mechanisms or causal effects of framing.
