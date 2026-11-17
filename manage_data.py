#!/usr/bin/env python3
"""
Data Management Utility for Crypto Filter
Provides commands to audit local parquet cache, check freshness, and safely sync symbols without hitting Binance rate limits.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
import pandas as pd

from src.data import BinanceFuturesFetcher, DataManager
from src.logger import logger

def audit_cache(data_dir: str = "data_cache", interval: str = "1h"):
    print(f"\n================ CACHE AUDIT ({interval}) ================")
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(f"_{interval}.parquet")]
    if not files:
        print(f"No files found for interval {interval}")
        return

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    stale_24h = []
    stale_7d = []
    stale_30d = []
    fresh = []
    errors = []

    for f in files:
        sym = f.replace(f"_{interval}.parquet", "")
        path = os.path.join(data_dir, f)
        try:
            df = pd.read_parquet(path, columns=["open_time", "close"])
            if df.empty:
                errors.append((sym, "Empty file"))
                continue
            last_ts = pd.to_datetime(df["open_time"].iloc[-1]).tz_localize(None)
            age_hours = (now_utc - last_ts).total_seconds() / 3600.0
            last_close = df["close"].iloc[-1]
            rows = len(df)

            item = (sym, last_ts, age_hours, last_close, rows)
            if age_hours <= 24:
                fresh.append(item)
            elif age_hours <= 168: # 7 days
                stale_24h.append(item)
            elif age_hours <= 720: # 30 days
                stale_7d.append(item)
            else:
                stale_30d.append(item)
        except Exception as e:
            errors.append((sym, str(e)))

    total = len(files)
    print(f"Total symbols cached ({interval}): {total}")
    print(f"  - Fresh (< 24h old):     {len(fresh)} ({len(fresh)/total*100:.1f}%)")
    print(f"  - Stale (> 24h - 7d):    {len(stale_24h)} ({len(stale_24h)/total*100:.1f}%)")
    print(f"  - Stale (> 7d - 30d):    {len(stale_7d)} ({len(stale_7d)/total*100:.1f}%)")
    print(f"  - Very Stale (> 30d):    {len(stale_30d)} ({len(stale_30d)/total*100:.1f}%)")
    if errors:
        print(f"  - Errors / Empty:        {len(errors)}")

    if stale_30d:
        print("\nSample of Very Stale files (>30 days old):")
        print(f"{'Symbol':<16} {'Last Candle Time (UTC)':<24} {'Age (Days)':<12} {'Rows':<8} {'Last Close'}")
        print("-" * 75)
        for sym, last_ts, age_h, last_close, rows in sorted(stale_30d, key=lambda x: x[2], reverse=True)[:15]:
            print(f"{sym:<16} {str(last_ts):<24} {age_h/24:<12.1f} {rows:<8} {last_close}")

    print("==========================================================\n")

def check_symbol(symbol: str, interval: str = "1h", data_dir: str = "data_cache"):
    symbol = symbol.upper()
    path = os.path.join(data_dir, f"{symbol}_{interval}.parquet")
    print(f"\n================ SYMBOL INSPECTION: {symbol} ({interval}) ================")
    if not os.path.exists(path):
        print(f"File '{path}' does not exist.")
        return

    try:
        df = pd.read_parquet(path)
        print(f"Rows:               {len(df)}")
        first_ts = df["open_time"].iloc[0]
        last_ts = df["open_time"].iloc[-1]
        print(f"First Candle:       {first_ts}")
        print(f"Last Candle:        {last_ts}")
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        age_hours = (now_utc - pd.to_datetime(last_ts).tz_localize(None)).total_seconds() / 3600.0
        print(f"Age of Last Candle: {age_hours:.1f} hours ({age_hours/24:.1f} days)")
        print(f"Last Close:         {df['close'].iloc[-1]}")

        for w in [40, 200]:
            if len(df) >= w:
                mean = df['close'].rolling(w).mean().iloc[-1]
                std = df['close'].rolling(w).std().iloc[-1]
                z = (df['close'].iloc[-1] - mean) / std if std > 0 else 0
                print(f"Z-Score (w={w}):     {z:+.6f} (mean: {mean:.5f}, std: {std:.5f})")
            else:
                print(f"Z-Score (w={w}):     Not enough data ({len(df)} < {w})")
    except Exception as e:
        print(f"Error reading symbol data: {e}")
    print("========================================================================\n")

def inspect_stats(symbol: str, period: str = "1h", limit: int = 30, output_csv: str = None):
    symbol = symbol.upper()
    fetcher = BinanceFuturesFetcher()
    print(f"\n================ HISTORICAL STATS: {symbol} (Period: {period}, Limit: {limit}) ================")
    
    # 1. Open Interest
    oi_data = fetcher.get_historical_stats_series(symbol, period, fetcher.OPEN_INTEREST_HIST, limit=limit)
    # 2. Top Trader Position Ratio
    pos_data = fetcher.get_historical_stats_series(symbol, period, fetcher.TOP_LS_POSITION, limit=limit)
    # 3. Top Trader Account Ratio
    acc_data = fetcher.get_historical_stats_series(symbol, period, fetcher.TOP_LS_ACCOUNT, limit=limit)

    if not oi_data and not pos_data and not acc_data:
        print(f"No statistical series data returned for {symbol}.")
        return

    # Merge into a clean DataFrame
    df_oi = pd.DataFrame(oi_data) if oi_data else pd.DataFrame()
    df_pos = pd.DataFrame(pos_data) if pos_data else pd.DataFrame()
    df_acc = pd.DataFrame(acc_data) if acc_data else pd.DataFrame()

    merged = None
    if not df_oi.empty:
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi = df_oi.rename(columns={'sumOpenInterest': 'oi_coins', 'sumOpenInterestValue': 'oi_usd'})
        merged = df_oi[['timestamp', 'oi_coins', 'oi_usd']]

    if not df_pos.empty:
        df_pos['timestamp'] = pd.to_datetime(df_pos['timestamp'], unit='ms')
        df_pos = df_pos.rename(columns={'longShortRatio': 'top_pos_ratio', 'longAccount': 'top_pos_long_pct', 'shortAccount': 'top_pos_short_pct'})
        pos_cols = df_pos[['timestamp', 'top_pos_ratio', 'top_pos_long_pct', 'top_pos_short_pct']]
        merged = pd.merge(merged, pos_cols, on='timestamp', how='outer') if merged is not None else pos_cols

    if not df_acc.empty:
        df_acc['timestamp'] = pd.to_datetime(df_acc['timestamp'], unit='ms')
        df_acc = df_acc.rename(columns={'longShortRatio': 'top_acc_ratio', 'longAccount': 'top_acc_long_pct', 'shortAccount': 'top_acc_short_pct'})
        acc_cols = df_acc[['timestamp', 'top_acc_ratio', 'top_acc_long_pct', 'top_acc_short_pct']]
        merged = pd.merge(merged, acc_cols, on='timestamp', how='outer') if merged is not None else acc_cols

    if merged is not None:
        merged = merged.sort_values('timestamp').reset_index(drop=True)
        # Format for clean terminal display
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        print(merged.tail(20).to_string(index=False))

        if output_csv:
            merged.to_csv(output_csv, index=False)
            print(f"\n[✓] Saved {len(merged)} records to {output_csv}")
    print("=========================================================================================\n")

def sync_symbols(symbols: list, interval: str = "1h", delay: float = 0.25):
    if BinanceFuturesFetcher.is_rate_limited():
        print(f"[!] Cannot sync: Binance rate limit cooldown active until {time.ctime(BinanceFuturesFetcher._rate_limited_until)}")
        return

    dm = DataManager()
    total = len(symbols)
    print(f"\nStarting polite sync for {total} symbols ({interval}) with {delay}s delay...")

    for i, sym in enumerate(symbols, 1):
        if BinanceFuturesFetcher.is_rate_limited():
            print(f"[!] Rate limit triggered during sync. Pausing at {sym} ({i}/{total}).")
            break
        try:
            print(f"[{i}/{total}] Syncing {sym} ({interval})...", end="", flush=True)
            df = dm.load_data(sym, interval, auto_sync=True)
            if df is not None and not df.empty:
                last_dt = df['open_time'].iloc[-1]
                last_c = df['close'].iloc[-1]
                print(f" OK ({len(df)} candles, last: {last_dt}, close: {last_c})")
            else:
                print(" FAILED (no data returned)")
        except Exception as e:
            print(f" ERROR: {e}")
        time.sleep(delay)

    print("Sync complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Crypto Filter Data Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Audit
    audit_parser = subparsers.add_parser("audit", help="Audit cache freshness")
    audit_parser.add_argument("--interval", default="1h", help="Interval to check (default: 1h)")

    # Check
    check_parser = subparsers.add_parser("check", help="Check details of a specific symbol")
    check_parser.add_argument("symbol", help="Symbol name (e.g. ZETAUSDT)")
    check_parser.add_argument("--interval", default="1h", help="Interval (default: 1h)")

    # Stats (Open Interest & Top Trader L/S)
    stats_parser = subparsers.add_parser("stats", help="Get historical Open Interest and Top Trader L/S ratios")
    stats_parser.add_argument("symbol", help="Symbol name (e.g. BTCUSDT)")
    stats_parser.add_argument("--period", default="1h", help="Period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)")
    stats_parser.add_argument("--limit", type=int, default=30, help="Number of candles (max 500)")
    stats_parser.add_argument("--save", help="Path to save output CSV (e.g. btc_oi_stats.csv)")

    # Sync
    sync_parser = subparsers.add_parser("sync", help="Synchronize symbol data from Binance")
    sync_parser.add_argument("--symbols", help="Comma-separated list of symbols (e.g. ZETAUSDT,COAIUSDT)")
    sync_parser.add_argument("--top", type=int, help="Sync top N volume symbols (e.g. 50)")
    sync_parser.add_argument("--interval", default="1h", help="Interval to sync (default: 1h)")
    sync_parser.add_argument("--delay", type=float, default=0.25, help="Delay between requests in seconds (default: 0.25)")

    args = parser.parse_args()

    if args.command == "audit":
        audit_cache(interval=args.interval)
    elif args.command == "check":
        check_symbol(args.symbol, interval=args.interval)
    elif args.command == "stats":
        inspect_stats(args.symbol, period=args.period, limit=args.limit, output_csv=args.save)
    elif args.command == "sync":
        symbols_to_sync = []
        if args.symbols:
            symbols_to_sync = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif args.top:
            dm = DataManager()
            symbols_to_sync = dm.get_universe(top_n=args.top)
        else:
            print("Please specify either --symbols or --top")
            sys.exit(1)
        sync_symbols(symbols_to_sync, interval=args.interval, delay=args.delay)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
