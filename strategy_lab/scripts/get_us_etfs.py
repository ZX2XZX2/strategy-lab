import io
import requests
import polars as pl

# URLs for symbol directories
NASDQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL = "http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def fetch_clean(url: str) -> str:
    txt = requests.get(url, timeout=30).text
    # Drop trailing timestamp line: "File Creation Time: mmddyyyyhhmm"
    return "\n".join(
        [ln for ln in txt.splitlines() if not ln.startswith("File Creation Time:")]
    )

# --- Load NASDAQ-listed ---
nasdaq_txt = fetch_clean(NASDQ_URL)
nasdaq = pl.read_csv(io.StringIO(nasdaq_txt), separator="|", has_header=True)
nasdaq = (
    nasdaq.filter((pl.col("ETF") == "Y") & (pl.col("Test Issue") == "N"))
    .with_columns(pl.lit("NASDAQ").alias("exchange"))
    .select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Security Name").alias("name"),
        pl.col("exchange"),
    )
)

# --- Load Other-listed (NYSE/Arca/American, etc.) ---
other_txt = fetch_clean(OTHER_URL)
other = pl.read_csv(io.StringIO(other_txt), separator="|", has_header=True)
other = (
    other.filter((pl.col("ETF") == "Y") & (pl.col("Test Issue") == "N"))
    .select(
        pl.col("ACT Symbol").alias("symbol"),
        pl.col("Security Name").alias("name"),
        pl.col("Exchange").alias("exchange"),
    )
)

# Combine & deduplicate
etfs = pl.concat([nasdaq, other]).unique(subset=["symbol"]).sort("symbol")

# --- Simple keyword-based category classification ---
categories = {
    "Technology": ["Tech", "Information Technology", "Semiconductor", "Software", "Cloud"],
    "Energy": ["Energy", "Oil", "Gas", "Petroleum"],
    "Financials": ["Financial", "Bank", "Insurance"],
    "Healthcare": ["Health", "Pharma", "Biotech", "Medical"],
    "Utilities": ["Utility"],
    "Real Estate": ["REIT", "Real Estate"],
    "Industrials": ["Industrial", "Aerospace", "Defense", "Manufacturing"],
    "Consumer": ["Consumer", "Retail", "Staples", "Discretionary"],
    "Materials": ["Materials", "Metals", "Mining", "Chemical"],
    "Communication": ["Communication", "Telecom", "Media"],
    "Government Bonds": ["Treasury", "Govt", "Government Bond"],
    "Corporate Bonds": ["Corporate Bond", "High Yield"],
    "International": ["Global", "International", "World", "Emerging", "China", "Japan", "Europe"],
}

def classify(name: str) -> str:
    if not isinstance(name, str):
        return "Other"
    for cat, kws in categories.items():
        for kw in kws:
            if kw.lower() in name.lower():
                return cat
    return "Other"

# Apply classifier
etfs = etfs.with_columns(
    pl.col("name").map_elements(classify, return_dtype=pl.Utf8).alias("category")
)

# Save to CSV
etfs.write_csv("us_etfs_with_category.csv")

print(etfs.head(10))
print(f"Saved {etfs.height} ETFs to us_etfs_with_category.csv")
