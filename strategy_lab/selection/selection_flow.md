# Stock Selection Flow - Indicator Calculator

### Overview

The stock selection pipeline executes the following steps:

1. **Download EOD Data**
   - Fetch historical data from the database for the specified date range.

2. **Preprocessing**
   - Apply split adjustments.
   - Organize data by ticker.

3. **Indicator Calculation**
   - Calculate activity over multiple windows.
   - Calculate intraday volatility.
   - Calculate relative strength over multiple windows.

4. **Fill Missing Values**
   - Fill nulls in larger window indicators using smaller window values.

5. **Clean Data**
   - Remove rows with null, NaN, or inf values.

6. **Sort & Save**
   - Sort by `date` and `ticker`.
   - Save as Parquet for efficient scanning.

7. **Rank, Bucket, Weighted Average**
   - Rank and bucketize indicators.
   - Calculate row-wise weighted averages (e.g., overall rank).

8. **Top N Selection (with ETF filtering)**
   - Select top N stocks per date with the highest overall rank.
   - If `selection:filter_out_etfs` is enabled in config.json:
     - Filter out ETFs from the selection.

### ASCII Flowchart
```
 ┌─────────────────────────────┐
 │    Download EOD Data        │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Apply Splits, Preprocess   │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │   Calculate Indicators      │
 │ (activity, volatility, RS)  │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │   Fill Missing Values       │
 │  (use smaller window data)  │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Clean Data (drop nulls,    │
 │  NaNs, infs)                │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Sort by Date and Ticker    │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Rank, Bucketize, Weighted  │
 │        Average Score        │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │ Select Top N per Date       │
 │ ─ If config.filter_out_etfs │
 │   → Exclude ETFs            │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │     Output Final Data       │
 └─────────────────────────────┘
 ```

### Output

A cleaned, ranked DataFrame with the top-performing stocks per date, ready for downstream use.
