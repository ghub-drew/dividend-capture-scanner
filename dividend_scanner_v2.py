"""
Dividend Capture Scanner  (v2)
Pulls upcoming ex-dividend dates from Yahoo Finance and ranks the best
dividend-capture candidates using four criteria:

  1. Stock quality  - prefer stable stocks you'd be comfortable holding
                      if the price doesn't rebound right away (low beta).
  2. Liquidity      - high average daily volume = easier to get in and out.
  3. Dividend yield - prioritise by annual yield tier:
                          Tier 1 : 8%+   (best)
                          Tier 2 : 7%+
                          Tier 3 : 5%+
                          Tier 4 : below 5% (only worth it if rebound is fast)
  4. Rebound time   - how many trading days the stock has historically taken
                      to climb back to its pre-ex-dividend price. Fewer days
                      = capital is freed up faster = better. This is the key
                      ranking metric.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── S&P 500 tickers (fetched from Wikipedia) ───
def get_sp500_tickers():
    import requests
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; dividend-scanner/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0]["Symbol"].tolist()

# ─── High-yield income names (mostly outside the S&P 500) ───
# These are where the 8%+ yields typically live: REITs, mortgage
# REITs, and business development companies (BDCs). Curated for liquidity.
HIGH_YIELD_TICKERS = [
    # Mortgage REITs (often 9-15% yield)
    "AGNC", "NLY", "STWD", "ABR", "RITM", "DX", "CIM", "TWO", "PMT", "ARI",
    "EFC", "MFA", "IVR", "ORC",
    # Business Development Companies (BDCs, often 8-12%)
    "ARCC", "MAIN", "FSK", "OBDC", "BXSL", "PSEC", "HTGC", "GBDC", "TSLX",
    "CSWC", "GAIN", "PFLT", "BBDC", "TCPC",
    # REITs (5-9% yield)
    "O", "WPC", "STAG", "NNN", "EPR", "GLPI", "OHI", "LTC", "SBRA", "ADC",
    "IRM", "VICI", "KRG", "GNL",
    # Energy infrastructure (6-8%)
    "KMI", "OKE", "WES", "ET", "EPD", "MPLX", "ENB",
    # Other high-yield blue chips
    "BTI", "VZ", "T", "MO", "PFE",
]

def get_universe():
    """Combine the S&P 500 with the curated high-yield list, de-duplicated."""
    sp500 = get_sp500_tickers()
    combined = list(dict.fromkeys(sp500 + HIGH_YIELD_TICKERS))  # preserves order, dedupes
    return combined

# ─── Filter config (adjust to taste) ───
MIN_ANNUAL_YIELD     = 0.03     # 3% floor. Tiers reward 5%/7%/8%+. Below 5%
                                # only ranks well if the rebound is very fast.
MIN_AVG_VOLUME       = 500_000  # shares/day - liquidity to enter/exit cleanly
MIN_STOCK_PRICE      = 10.0     # avoid penny-stock volatility
MAX_STOCK_PRICE      = 500.0    # keep position sizes manageable
EX_DATE_WINDOW_DAYS  = 14       # only show stocks with ex-date in next N days

# ─── Rebound-metric config ───
REBOUND_LOOKBACK_EVENTS = 6     # measure the last N ex-dividend events
REBOUND_MAX_DAYS        = 60    # cap the search window per event (trading days)


def yield_tier(annual_yield: float) -> int:
    """Classify a stock by annual dividend yield (yield tiers)."""
    if annual_yield >= 0.08:
        return 1
    if annual_yield >= 0.07:
        return 2
    if annual_yield >= 0.05:
        return 3
    return 4


def compute_rebound(hist: pd.DataFrame):
    """
    Estimate how long the stock historically takes to recover to its
    pre-ex-dividend price after the price drops on the ex-date.

    Returns (avg_rebound_days, success_rate, events_measured).
      - avg_rebound_days : average trading days to climb back to par
                           (only over events that DID recover)
      - success_rate     : fraction of measured events that recovered
                           within REBOUND_MAX_DAYS
      - events_measured  : how many past ex-dates we were able to test
    Returns (None, None, 0) if there isn't enough history.
    """
    if hist is None or hist.empty or "Dividends" not in hist.columns:
        return None, None, 0

    closes = hist["Close"]
    ex_dates = list(hist.index[hist["Dividends"] > 0])
    if not ex_dates:
        return None, None, 0

    # Most recent events first, limited to the lookback window
    ex_dates = ex_dates[-REBOUND_LOOKBACK_EVENTS:]

    rebound_days = []
    recovered_count = 0
    measured = 0

    for ex_date in ex_dates:
        # Price the day BEFORE the ex-date = the price you'd pay to capture
        pre = closes[closes.index < ex_date]
        if pre.empty:
            continue
        pre_price = pre.iloc[-1]

        # Prices from the ex-date forward, capped to the search window
        post = closes[closes.index >= ex_date].iloc[:REBOUND_MAX_DAYS + 1]
        if len(post) < 2:
            continue

        measured += 1

        # First day the price gets back to (or above) the pre-ex price
        recovered = post[post >= pre_price]
        if recovered.empty:
            continue  # never recovered within the window
        recovery_date = recovered.index[0]
        days = post.index.get_loc(recovery_date)  # trading days since ex-date
        rebound_days.append(days)
        recovered_count += 1

    if measured == 0:
        return None, None, 0

    avg_rebound = (sum(rebound_days) / len(rebound_days)) if rebound_days else None
    success_rate = recovered_count / measured
    return avg_rebound, success_rate, measured


def score(row: dict) -> float:
    """
    Composite score honouring the strategy priorities:
      - Yield tier is the strongest factor (8%+ best).
      - Faster rebound adds points (frees up capital sooner).
      - Consistent recovery adds points.
      - Liquidity adds a little; high beta (instability) costs a little.
    A low-yield stock with a very fast, reliable rebound can still climb.
    """
    tier_score = {1: 100, 2: 85, 3: 65, 4: 35}[row["_tier"]]

    rebound = row["_rebound_days"]
    if rebound is None:
        rebound_score = 0          # unknown rebound = no bonus
    else:
        rebound_score = max(0, 30 - rebound)   # 0 days -> 30, 30+ days -> 0

    success = row["_success_rate"]
    recovery_bonus = (success * 10) if success is not None else 0

    liquidity_score = min(10, row["_avg_volume"] / 1_000_000 * 2)
    beta_penalty = row["_beta"] * 2

    return tier_score + rebound_score + recovery_bonus + liquidity_score - beta_penalty


def fetch_candidate(ticker: str, today: datetime) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- Cheap filters first (from info) ---
        ex_date_ts = info.get("exDividendDate")
        if not ex_date_ts:
            return None
        ex_date = datetime.fromtimestamp(ex_date_ts)
        days_until = (ex_date - today).days
        if not (0 < days_until <= EX_DATE_WINDOW_DAYS):
            return None

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
            return None

        div_rate = info.get("dividendRate")   # annual dividend in $
        if not div_rate:
            return None
        annual_yield = div_rate / price
        if annual_yield < MIN_ANNUAL_YIELD:
            return None

        avg_volume = info.get("averageVolume") or 0
        if avg_volume < MIN_AVG_VOLUME:
            return None

        # --- Expensive step: only for stocks that passed the cheap filters ---
        hist = stock.history(period="2y", auto_adjust=False)

        # Infer payment frequency from the last year of dividend events
        freq = 4  # default quarterly
        if "Dividends" in hist.columns:
            one_year_ago = hist.index.max() - pd.Timedelta(days=365)
            recent_events = hist.index[(hist["Dividends"] > 0) & (hist.index >= one_year_ago)]
            if len(recent_events) > 0:
                freq = len(recent_events)
        div_per_payment = div_rate / freq
        capture_yield = div_per_payment / price

        avg_rebound, success_rate, events = compute_rebound(hist)

        beta = info.get("beta") or 1.0
        sector = info.get("sector", "Unknown")
        name = info.get("shortName", ticker)
        tier = yield_tier(annual_yield)

        return {
            "Ticker":        ticker,
            "Name":          name,
            "Sector":        sector,
            "Price":         round(price, 2),
            "Ex-Date":       ex_date.strftime("%Y-%m-%d"),
            "Days Until":    days_until,
            "Tier":          tier,
            "Annual Yield":  f"{annual_yield:.2%}",
            "Capture Yield": f"{capture_yield:.2%}",
            "Div/Capture":   round(div_per_payment, 4),
            "Rebound Days":  round(avg_rebound, 1) if avg_rebound is not None else "N/A",
            "Recovered":     f"{success_rate:.0%}" if success_rate is not None else "N/A",
            "Avg Volume":    f"{avg_volume:,.0f}",
            "Beta":          round(beta, 2),
            # private fields used only for scoring
            "_tier":         tier,
            "_rebound_days": avg_rebound,
            "_success_rate": success_rate,
            "_avg_volume":   avg_volume,
            "_beta":         beta,
        }
    except Exception:
        return None


def run_scanner():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"\nDividend Capture Scanner - {today.strftime('%Y-%m-%d')}")
    print(f"Looking for ex-dates within the next {EX_DATE_WINDOW_DAYS} days\n")

    print("Fetching S&P 500 tickers and high-yield income names...")
    tickers = get_universe()
    print(f"Scanning {len(tickers)} stocks (S&P 500 + high-yield REITs/BDCs)...")
    print("(Candidates that pass the basic filters also get a rebound-time")
    print(" analysis from 2 years of history, so this can take a few minutes.)\n")

    candidates = []
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} scanned, {len(candidates)} candidates so far...")
        result = fetch_candidate(ticker, today)
        if result:
            candidates.append(result)

    if not candidates:
        print("\nNo candidates found matching the current filters.")
        return

    df = pd.DataFrame(candidates)
    df["Score"] = df.apply(score, axis=1).round(1)

    # Sort best-first by score (which already bakes in tier + rebound + etc.)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index += 1

    # Columns to display (drop the private _fields)
    display_cols = [
        "Ticker", "Name", "Sector", "Price", "Ex-Date", "Days Until",
        "Tier", "Annual Yield", "Capture Yield", "Div/Capture",
        "Rebound Days", "Recovered", "Avg Volume", "Beta", "Score",
    ]
    shown = df[display_cols]

    print(f"\n{'='*100}")
    print(f"  TOP DIVIDEND CAPTURE CANDIDATES  ({len(df)} found)")
    print(f"{'='*100}")
    print(shown.to_string())

    print(f"\nYield tiers (annual):  1 = 8%+   2 = 7%+   3 = 5%+   4 = below 5%")
    print(f"Rebound Days = avg trading days to recover to the pre-ex price (lower is better)")
    print(f"Recovered    = how often it recovered within {REBOUND_MAX_DAYS} trading days")
    print(f"\nFilters applied:")
    print(f"  Min annual yield  : {MIN_ANNUAL_YIELD:.0%}")
    print(f"  Min avg volume    : {MIN_AVG_VOLUME:,} shares/day")
    print(f"  Price range       : ${MIN_STOCK_PRICE} - ${MAX_STOCK_PRICE}")
    print(f"  Ex-date window    : next {EX_DATE_WINDOW_DAYS} days")

    out_file = f"candidates_{today.strftime('%Y%m%d')}.csv"
    shown.to_csv(out_file, index=True)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    run_scanner()
