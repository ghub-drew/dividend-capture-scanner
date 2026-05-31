"""
Dividend Capture Scanner
Pulls upcoming ex-dividend dates from Yahoo Finance and filters
for the best capture candidates based on yield, volume, and price.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# --- S&P 500 tickers (fetched from Wikipedia) ---
def get_sp500_tickers():
    import requests
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; dividend-scanner/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0]["Symbol"].tolist()

# --- Filter config (adjust these to match Rick's criteria) ---
MIN_DIVIDEND_YIELD   = 0.005   # 0.5% minimum dividend yield per payment
MIN_AVG_VOLUME       = 500_000  # shares/day — need liquidity to enter/exit cleanly
MIN_STOCK_PRICE      = 10.0    # avoid penny-stock volatility
MAX_STOCK_PRICE      = 500.0   # keep position sizes manageable
EX_DATE_WINDOW_DAYS  = 14      # only show stocks with ex-date in next N days

def fetch_candidate(ticker: str, today: datetime) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        ex_date_ts = info.get("exDividendDate")
        if not ex_date_ts:
            return None

        ex_date = datetime.fromtimestamp(ex_date_ts)
        days_until = (ex_date - today).days
        if not (0 < days_until <= EX_DATE_WINDOW_DAYS):
            return None

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return None
        if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
            return None

        div_rate = info.get("dividendRate")
        frequency = info.get("dividendFrequency") or 4  # assume quarterly
        if not div_rate:
            return None
        div_per_payment = div_rate / frequency
        div_yield_per_payment = div_per_payment / price
        if div_yield_per_payment < MIN_DIVIDEND_YIELD:
            return None

        avg_volume = info.get("averageVolume") or 0
        if avg_volume < MIN_AVG_VOLUME:
            return None

        beta = info.get("beta") or 1.0
        sector = info.get("sector", "Unknown")
        name = info.get("shortName", ticker)

        return {
            "Ticker":        ticker,
            "Name":          name,
            "Sector":        sector,
            "Price":         round(price, 2),
            "Ex-Date":       ex_date.strftime("%Y-%m-%d"),
            "Days Until":    days_until,
            "Div/Share":     round(div_per_payment, 4),
            "Yield/Payment": f"{div_yield_per_payment:.2%}",
            "Avg Volume":    f"{avg_volume:,.0f}",
            "Beta":          round(beta, 2),
        }
    except Exception:
        return None


def score(row: dict) -> float:
    """Higher yield and more days remaining = higher score."""
    yield_val = float(row["Yield/Payment"].strip("%")) / 100
    days = row["Days Until"]
    beta = row["Beta"]
    # Reward yield, reward having time left, penalise high beta
    return yield_val * 1000 + days * 0.5 - beta * 2


def run_scanner():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"\nDividend Capture Scanner — {today.strftime('%Y-%m-%d')}")
    print(f"Looking for ex-dates within the next {EX_DATE_WINDOW_DAYS} days\n")

    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} stocks...\n")

    candidates = []
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} scanned, {len(candidates)} candidates so far...")
        result = fetch_candidate(ticker, today)
        if result:
            candidates.append(result)

    if not candidates:
        print("No candidates found matching the current filters.")
        return

    df = pd.DataFrame(candidates)
    df["_score"] = df.apply(score, axis=1)
    df = df.sort_values("_score", ascending=False).drop(columns=["_score"])
    df = df.reset_index(drop=True)
    df.index += 1

    print(f"\n{'='*80}")
    print(f"  TOP DIVIDEND CAPTURE CANDIDATES  ({len(df)} found)")
    print(f"{'='*80}")
    print(df.to_string())
    print(f"\nFilters applied:")
    print(f"  Min yield/payment : {MIN_DIVIDEND_YIELD:.1%}")
    print(f"  Min avg volume    : {MIN_AVG_VOLUME:,} shares/day")
    print(f"  Price range       : ${MIN_STOCK_PRICE} – ${MAX_STOCK_PRICE}")
    print(f"  Ex-date window    : next {EX_DATE_WINDOW_DAYS} days")

    # Save to CSV for reference
    out_file = f"candidates_{today.strftime('%Y%m%d')}.csv"
    df.to_csv(out_file, index=True)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    run_scanner()
