"""
Data Engine for Crypto Filter
Handles fetching from Binance Futures and local caching via Parquet.
"""
import requests
import pandas as pd
import numpy as np
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Import from config
from src.config import MANDATORY_CRYPTO, BENCHMARK_SYMBOL, TIMEZONE_OFFSET, IGNORED_CRYPTO

logger = logging.getLogger(__name__)

def get_default_fetch_days(interval: str, years: float = 0) -> int:
    """Returns number of historical days to fetch to ensure sufficient warmup bars (at least 3,000 to 5,000 candles)."""
    if years > 0:
        return int(365 * years)
    mapping = {
        '1m': 14,      # ~20,000 candles
        '3m': 21,      # ~10,000 candles
        '5m': 30,      # ~8,640 candles
        '15m': 90,     # ~8,640 candles
        '30m': 180,    # ~8,640 candles
        '1h': 200,     # ~4,800 candles
        '2h': 400,     # ~4,800 candles
        '4h': 750,     # ~4,500 candles (~2 years)
        '6h': 1000,    # ~4,000 candles
        '8h': 1200,    # ~3,600 candles
        '12h': 1500,   # ~3,000 candles
        '1d': 1825,    # ~1,825 candles (5 years)
        '3d': 3650,    # 10 years
        '1w': 3650,    # 10 years
        '1M': 3650     # 10 years
    }
    return mapping.get(interval, 200)

def get_yf_max_days(interval: str, years: float = 0) -> int:
    """Returns the maximum allowed lookback days for Yahoo Finance API by interval."""
    if interval == '1m':
        return 7
    elif interval in ['2m', '3m', '5m', '15m', '30m']:
        return 59
    elif interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
        return min(int(365 * years) if years > 0 else 720, 720)
    else: # 1d, 3d, 1w, 1M
        return int(365 * years) if years > 0 else 3650

class BinanceFuturesFetcher:
    """Fetches data from Binance Perpetual Futures (USDT-M)"""
    
    BASE_URLS = [
        os.environ.get("BINANCE_BASE_URL", "https://fapi.binance.com"),
        "https://fapi1.binance.com",
        "https://fapi2.binance.com",
        "https://fapi3.binance.com",
        "https://fapi4.binance.com"
    ]
    BASE_URL = BASE_URLS[0]
    TICKER_24H = "/fapi/v1/ticker/24hr"
    KLINES = "/fapi/v1/klines"
    EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
    ORDERBOOK = "/fapi/v1/depth"
    PREMIUM_INDEX = "/fapi/v1/premiumIndex"
    
    SYMBOL_ADL_RISK = "/fapi/v1/symbolAdlRisk"
    OPEN_INTEREST_HIST = "/futures/data/openInterestHist"
    TOP_LS_POSITION = "/futures/data/topLongShortPositionRatio"
    TOP_LS_ACCOUNT = "/futures/data/topLongShortAccountRatio"
    GLOBAL_LS_ACCOUNT = "/futures/data/globalLongShortAccountRatio"
    TAKER_BUY_SELL = "/futures/data/takerlongshortRatio"
    
    # Class-level caches shared across all instances
    _adl_risk_cache = {}
    _adl_risk_last_fetch = 0.0
    _stats_cache = {}
    _exchange_info_cache = None
    _exchange_info_last_fetch = 0.0
    _universe_meta_cache = None
    _universe_meta_last_fetch = 0.0
    _cache_expiry = 300.0  # 5 minutes cache
    _cache_lock = threading.Lock()
    _rate_limited_until = 1794804260.0  # Cooldown timestamp for Binance 418/429
    
    @classmethod
    def is_rate_limited(cls) -> bool:
        return time.time() < cls._rate_limited_until

    def __init__(self, rate_limit_delay: float = 0.1):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json"
        })
        self.rate_limit_delay = rate_limit_delay

    def get_all_adl_risks(self) -> dict:
        """Fetch all symbol ADL risks in one batch and return a map of {symbol: numeric_risk}"""
        now = time.time()
        with BinanceFuturesFetcher._cache_lock:
            if BinanceFuturesFetcher._adl_risk_cache and (now - BinanceFuturesFetcher._adl_risk_last_fetch < BinanceFuturesFetcher._cache_expiry):
                return BinanceFuturesFetcher._adl_risk_cache
            
        try:
            data = self._request(self.SYMBOL_ADL_RISK)
            if data and isinstance(data, list):
                risk_map = {}
                mapping = {"low": 0, "medium": 1, "high": 2}
                for item in data:
                    sym = item.get("symbol")
                    risk_str = item.get("adlRisk", "").lower()
                    risk_map[sym] = mapping.get(risk_str, 0)
                
                with BinanceFuturesFetcher._cache_lock:
                    BinanceFuturesFetcher._adl_risk_cache = risk_map
                    BinanceFuturesFetcher._adl_risk_last_fetch = now
        except Exception as e:
            logger.error(f"Error fetching ADL risks: {e}")
            
        with BinanceFuturesFetcher._cache_lock:
            return BinanceFuturesFetcher._adl_risk_cache

    def get_historical_stats(self, symbol: str, period: str, endpoint: str) -> dict:
        """Generic method to fetch and cache historical stats endpoints with limit=1"""
        now = time.time()
        cache_key = (symbol, period, endpoint)
        with BinanceFuturesFetcher._cache_lock:
            if cache_key in BinanceFuturesFetcher._stats_cache:
                ts, val = BinanceFuturesFetcher._stats_cache[cache_key]
                if now - ts < BinanceFuturesFetcher._cache_expiry:
                    return val
                
        try:
            params = {"symbol": symbol, "period": period, "limit": 1}
            data = self._request(endpoint, params=params)
            
            res = {}
            if data and isinstance(data, list) and len(data) > 0:
                res = data[-1]  # Get latest item
                
            with BinanceFuturesFetcher._cache_lock:
                BinanceFuturesFetcher._stats_cache[cache_key] = (now, res)
            return res
        except Exception as e:
            logger.error(f"Error fetching {endpoint} for {symbol} ({period}): {e}")
            return {}

    def get_historical_stats_series(self, symbol: str, period: str, endpoint: str, limit: int = 30) -> list:
        """Fetch and cache historical series for Binance statistics endpoints"""
        now = time.time()
        cache_key = ("series", symbol, period, endpoint, limit)
        with BinanceFuturesFetcher._cache_lock:
            if cache_key in BinanceFuturesFetcher._stats_cache:
                ts, val = BinanceFuturesFetcher._stats_cache[cache_key]
                if now - ts < BinanceFuturesFetcher._cache_expiry:
                    return val

        try:
            params = {"symbol": symbol, "period": period, "limit": limit}
            data = self._request(endpoint, params=params)
            res = data if isinstance(data, list) else []
            if res:
                with BinanceFuturesFetcher._cache_lock:
                    BinanceFuturesFetcher._stats_cache[cache_key] = (now, res)
            return res
        except Exception as e:
            logger.error(f"Error fetching {endpoint} series for {symbol}: {e}")
            return []
            
    def get_historical_funding_rate(self, symbol: str, limit: int = 30) -> list:
        """Fetch and cache historical funding rates for a symbol"""
        now = time.time()
        cache_key = ("funding_series", symbol, limit)
        with BinanceFuturesFetcher._cache_lock:
            if cache_key in BinanceFuturesFetcher._stats_cache:
                ts, val = BinanceFuturesFetcher._stats_cache[cache_key]
                if now - ts < BinanceFuturesFetcher._cache_expiry:
                    return val

        try:
            params = {"symbol": symbol, "limit": limit}
            data = self._request("/fapi/v1/fundingRate", params=params)
            res = data if isinstance(data, list) else []
            if res:
                with BinanceFuturesFetcher._cache_lock:
                    BinanceFuturesFetcher._stats_cache[cache_key] = (now, res)
            return res
        except Exception as e:
            logger.error(f"Error fetching funding rate series for {symbol}: {e}")
            return []

    def get_source_series(self, symbol: str, source: str, interval: str = "1h", limit: int = 200) -> pd.Series:
        """Fetch a clean time-indexed pd.Series for non-OHLCV market data sources (with disk+mem cache)."""
        if symbol in ['QQQUSDT', 'SPYUSDT', 'XAUUSDT', 'XAGUSDT']:
            return pd.Series(dtype=float)

        period_mapping = {
            '1m': '5m', '3m': '5m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '6h',
            '12h': '12h', '1d': '1d', '3d': '1d', '1w': '1d'
        }
        period = period_mapping.get(interval, '1h')
        safe_limit = min(max(int(limit), 30), 500)

        from src.cache_store import SeriesDiskCache
        if not hasattr(BinanceFuturesFetcher, "_series_disk_cache"):
            BinanceFuturesFetcher._series_disk_cache = SeriesDiskCache()
        cached_s = BinanceFuturesFetcher._series_disk_cache.get(symbol, source, period)
        if cached_s is not None and len(cached_s) >= min(safe_limit, 30):
            return cached_s

        source_map = {
            'oi_usd': (self.OPEN_INTEREST_HIST, 'sumOpenInterestValue', 'timestamp'),
            'oi_coins': (self.OPEN_INTEREST_HIST, 'sumOpenInterest', 'timestamp'),
            'top_ls_pos': (self.TOP_LS_POSITION, 'longShortRatio', 'timestamp'),
            'top_ls_acc': (self.TOP_LS_ACCOUNT, 'longShortRatio', 'timestamp'),
            'global_ls': (self.GLOBAL_LS_ACCOUNT, 'longShortRatio', 'timestamp'),
            'taker_bs': (self.TAKER_BUY_SELL, 'buySellRatio', 'timestamp'),
        }

        if source == 'funding':
            raw = self.get_historical_funding_rate(symbol, limit=safe_limit)
            val_key, ts_key = 'fundingRate', 'fundingTime'
        elif source in source_map:
            endpoint, val_key, ts_key = source_map[source]
            raw = self.get_historical_stats_series(symbol, period, endpoint, limit=safe_limit)
        else:
            return pd.Series(dtype=float)

        if not raw:
            return pd.Series(dtype=float)

        try:
            df = pd.DataFrame(raw)
            if val_key not in df.columns or ts_key not in df.columns:
                return pd.Series(dtype=float)
            df[ts_key] = pd.to_datetime(df[ts_key], unit='ms')
            df[val_key] = pd.to_numeric(df[val_key], errors='coerce')
            df = df.dropna(subset=[val_key]).sort_values(ts_key).drop_duplicates(subset=[ts_key])
            s = pd.Series(df[val_key].values, index=df[ts_key])
            BinanceFuturesFetcher._series_disk_cache.set(symbol, source, period, s)
            return s
        except Exception as e:
            logger.debug(f"Error parsing {source} series for {symbol}: {e}")
            return pd.Series(dtype=float)

    def _request(self, endpoint: str, params: dict = None) -> Union[dict, list]:
        if time.time() < BinanceFuturesFetcher._rate_limited_until:
            wait_sec = int(BinanceFuturesFetcher._rate_limited_until - time.time())
            if wait_sec > 0 and wait_sec % 60 == 0:
                logger.warning(f"Binance rate limit cooldown active: skipping {endpoint} (resumes in {wait_sec}s)")
            return {}

        last_error = None
        for base in self.BASE_URLS:
            url = f"{base}{endpoint}"
            try:
                response = self.session.get(url, params=params, timeout=8)
                response.raise_for_status()
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                return response.json()
            except requests.exceptions.HTTPError as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code in (418, 429):
                    cooldown = 180  # Default 3 mins
                    try:
                        resp_json = e.response.json()
                        msg = resp_json.get("msg", "")
                        if "banned until" in msg:
                            import re
                            match = re.search(r"banned until\s+(\d+)", msg)
                            if match:
                                ban_ts = int(match.group(1)) / 1000.0
                                cooldown = max(10, int(ban_ts - time.time()) + 3)
                    except Exception:
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                cooldown = int(retry_after) + 3
                            except ValueError:
                                pass
                    BinanceFuturesFetcher._rate_limited_until = time.time() + cooldown
                    logger.critical(f"Binance IP rate-limited / banned (HTTP {status_code}). Backing off for {cooldown}s until {time.ctime(BinanceFuturesFetcher._rate_limited_until)}.")
                    # All Binance mirrors share IP ban; immediately abort to avoid prolonging ban
                    return {}

                logger.warning(f"HTTP Error {status_code} for {url}: {e}. Retrying with mirror...")
                last_error = e
                continue
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                logger.warning(f"Connection/DNS failed for {url}: {e}. Retrying with mirror...")
                continue
            except (requests.exceptions.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(f"Invalid JSON/empty response from {url}: {e}. Retrying with mirror...")
                continue
            except Exception as e:
                logger.error(f"Request failed {url}: {e}")
                return {}
        if last_error:
            logger.error(f"All Binance endpoints failed for {endpoint}: {last_error}")
        return {}

    def get_orderbooks(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Get and cache orderbook for a single symbol (180s TTL)"""
        now = time.time()
        cache_key = ("orderbook", symbol, limit)
        with BinanceFuturesFetcher._cache_lock:
            if cache_key in BinanceFuturesFetcher._stats_cache:
                ts, val = BinanceFuturesFetcher._stats_cache[cache_key]
                if now - ts < 180.0:
                    return val

        data = self._request(self.ORDERBOOK, {'symbol': symbol, 'limit': limit})
        if not data: return pd.DataFrame()
        
        bids = pd.DataFrame(data.get('bids', []), columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(data.get('asks', []), columns=['price', 'qty'], dtype=float)
        
        bids['side'] = 'bid'
        asks['side'] = 'ask'
        res = pd.concat([bids, asks])
        with BinanceFuturesFetcher._cache_lock:
            BinanceFuturesFetcher._stats_cache[cache_key] = (now, res)
        return res

    def get_funding_rates(self) -> dict:
        """Get latest funding rates for all symbols"""
        data = self._request(self.PREMIUM_INDEX)
        if not data: return {}
        return {item['symbol']: float(item.get('lastFundingRate', 0)) for item in data}

    def get_24h_changes(self) -> dict:
        """Get and cache 24h priceChangePercent for all symbols (60s TTL)"""
        now = time.time()
        cache_key = ("ticker_24h_chg",)
        with BinanceFuturesFetcher._cache_lock:
            if cache_key in BinanceFuturesFetcher._stats_cache:
                ts, val = BinanceFuturesFetcher._stats_cache[cache_key]
                if now - ts < 60.0:
                    return val

        data = self._request(self.TICKER_24H)
        if not data or not isinstance(data, list):
            return {}
        res = {}
        for item in data:
            sym = item.get('symbol')
            if sym:
                try:
                    res[sym] = float(item.get('priceChangePercent', 0.0))
                except (ValueError, TypeError):
                    pass
        with BinanceFuturesFetcher._cache_lock:
            BinanceFuturesFetcher._stats_cache[cache_key] = (now, res)
        return res

    def get_books_status(self, symbol: str, impact_usd: float = 1000, imbalance_pct: float = 0.05, limit: int = 500):
        """
        Calculate impact spread and orderbook imbalance.
        impact_usd: USD size to calculate average execution price.
        imbalance_pct: range around mid-price to calculate volume imbalance.
        """
        df = self.get_orderbooks(symbol, limit=limit)
        if df.empty: return {}

        bids = df[df["side"] == "bid"].sort_values("price", ascending=False)
        asks = df[df["side"] == "ask"].sort_values("price", ascending=True)
        if bids.empty or asks.empty: return {}

        mid = (bids.iloc[0]["price"] + asks.iloc[0]["price"]) / 2

        def calc_impact_price(book, target_usd):
            notional = 0.0
            total_qty = 0.0
            for _, row in book.iterrows():
                p, q = row["price"], row["qty"]
                v = p * q
                if notional + v >= target_usd:
                    needed_v = target_usd - notional
                    total_qty += needed_v / p
                    notional = target_usd
                    break
                total_qty += q
                notional += v
            return target_usd / total_qty if notional >= target_usd and total_qty > 0 else None

        avg_ask = calc_impact_price(asks, impact_usd)
        avg_bid = calc_impact_price(bids, impact_usd)
        
        impact_spread = (avg_ask - avg_bid) / mid if (avg_ask and avg_bid) else None
        
        # Imbalance
        lower = mid * (1 - imbalance_pct)
        upper = mid * (1 + imbalance_pct)
        bid_liq = bids[bids.price >= lower]["qty"].sum() * mid
        ask_liq = asks[asks.price <= upper]["qty"].sum() * mid
        total_liq = bid_liq + ask_liq
        imbalance = (bid_liq - ask_liq) / total_liq if total_liq > 0 else None

        return {
            "mid_price": mid,
            "impact_spread": impact_spread,
            "orderbook_imbalance": imbalance,
            "bid_liquidity": bid_liq,
            "ask_liquidity": ask_liq
        }

    def get_exchange_info(self) -> dict:
        """Fetch and cache Binance exchange info."""
        now = time.time()
        with BinanceFuturesFetcher._cache_lock:
            if BinanceFuturesFetcher._exchange_info_cache and (now - BinanceFuturesFetcher._exchange_info_last_fetch < BinanceFuturesFetcher._cache_expiry):
                return BinanceFuturesFetcher._exchange_info_cache
        
        info = self._request(self.EXCHANGE_INFO)
        if info:
            with BinanceFuturesFetcher._cache_lock:
                BinanceFuturesFetcher._exchange_info_cache = info
                BinanceFuturesFetcher._exchange_info_last_fetch = now
        return info or {}

    def get_universe_metadata(self) -> dict:
        """Fetch all unique underlying types and subtypes from exchange info with caching"""
        now = time.time()
        with BinanceFuturesFetcher._cache_lock:
            if BinanceFuturesFetcher._universe_meta_cache and (now - BinanceFuturesFetcher._universe_meta_last_fetch < BinanceFuturesFetcher._cache_expiry):
                return BinanceFuturesFetcher._universe_meta_cache

        info = self.get_exchange_info()
        if not info or 'symbols' not in info:
            return {"types": [], "subtypes": [], "symbol_meta": {}}
            
        types = set()
        subtypes = set()
        symbol_meta = {}
        has_empty_subtypes = False
        
        for s in info.get('symbols', []):
            if s.get('status') == 'TRADING' and s.get('contractType') == 'PERPETUAL' and s.get('symbol', '').endswith('USDT'):
                sym = s['symbol']
                t = s.get('underlyingType')
                st = s.get('underlyingSubType', [])
                
                if t: types.add(t)
                if not st or len(st) == 0:
                    has_empty_subtypes = True
                else:
                    for sub in st:
                        subtypes.add(sub)
                    
                symbol_meta[sym] = {
                    "type": t,
                    "subtypes": st
                }
                
        subtypes_list = sorted(list(subtypes))
        if has_empty_subtypes:
            subtypes_list.insert(0, "-")
            
        result = {
            "types": sorted(list(types)),
            "subtypes": subtypes_list,
            "symbol_meta": symbol_meta
        }
        with BinanceFuturesFetcher._cache_lock:
            BinanceFuturesFetcher._universe_meta_cache = result
            BinanceFuturesFetcher._universe_meta_last_fetch = now

        return result

    def get_top_volume_symbols_with_meta(self, top_n: int = 50, exclude: List[str] = None, bottom: bool = False, types: List[str] = None, subtypes: List[str] = None) -> List[dict]:
        """Get top N symbols with volume and metadata, optionally filtered by types/subtypes"""
        if exclude is None:
            exclude = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USTUSDT', 'FDUSDUSDT']
            
        # Add IGNORED_CRYPTO to exclude list
        if IGNORED_CRYPTO:
            exclude = list(set(exclude + IGNORED_CRYPTO))
            
        data = self._request(self.TICKER_24H)
        if not data: return []
        
        meta_res = self.get_universe_metadata()
        meta = meta_res.get("symbol_meta", {})
        
        df = pd.DataFrame(data)
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce').fillna(0.0)
        df = df.sort_values('quoteVolume', ascending=bottom)
        
        results = []
        for _, row in df.iterrows():
            sym = row['symbol']
            if not sym.endswith('USDT') or sym in exclude:
                continue
                
            s_meta = meta.get(sym, {})
            s_type = s_meta.get('type') or 'COIN'
            s_subs = s_meta.get('subtypes', [])
            
            # Check types match (if types filter provided)
            if types and s_type not in types:
                continue
                
            # Check subtypes match (if subtypes filter provided)
            if subtypes:
                if not s_subs or len(s_subs) == 0:
                    has_sub = ("-" in subtypes)
                else:
                    has_sub = any(sub in subtypes for sub in s_subs)
                if not has_sub:
                    continue
            
            vol = float(row['quoteVolume'])
            if vol >= 1e9:
                vol_fmt = f"${vol/1e9:.2f}B"
            elif vol >= 1e6:
                vol_fmt = f"${vol/1e6:.2f}M"
            elif vol >= 1e3:
                vol_fmt = f"${vol/1e3:.2f}K"
            else:
                vol_fmt = f"${vol:.0f}"

            results.append({
                "symbol": sym,
                "type": s_type,
                "subtypes": s_subs,
                "volume": vol,
                "volume_formatted": vol_fmt
            })
            if len(results) >= int(top_n):
                break
                
        return results

    def get_top_volume_symbols(self, top_n: int = 50, exclude: List[str] = None, bottom: bool = False, types: List[str] = None, subtypes: List[str] = None) -> List[str]:
        """Get top N symbols by 24h Quote (USDT) Volume, optionally filtered by types/subtypes"""
        items = self.get_top_volume_symbols_with_meta(top_n=top_n, exclude=exclude, bottom=bottom, types=types, subtypes=subtypes)
        return [i['symbol'] for i in items]

    def get_all_symbols(self) -> List[str]:
        """Get all valid USDT perpetual symbols"""
        info = self.get_exchange_info()
        if not info or 'symbols' not in info:
            return []
            
        symbols = []
        for s in info['symbols']:
            sym = s['symbol']
            if sym.endswith('USDT') and s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL':
                if sym not in IGNORED_CRYPTO:
                    symbols.append(sym)
        return sorted(symbols)

    def fetch_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = None) -> pd.DataFrame:
        """Fetch klines for a single symbol"""
        if symbol in ['QQQUSDT', 'SPYUSDT']:
            return _fetch_yf_data(symbol, interval, start_ts=start_time, end_ts=end_time, limit=limit)
            
        params = {'symbol': symbol, 'interval': interval}
        
        if start_time: params['startTime'] = start_time
        if end_time: params['endTime'] = end_time
        if limit: params['limit'] = min(limit, 1500)
        
        raw = self._request(self.KLINES, params)
        if not raw: return pd.DataFrame()
        
        # standard binance kline columns
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']
        
        df = pd.DataFrame(raw, columns=cols)
        
        # Optimize types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float64)
            
        df = df.sort_values('open_time')
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]

    def fetch_history(self, 
                      symbol: str, 
                      interval: str, 
                      years: float = 0,
                      start_time: Union[str, int, datetime, None] = None, 
                      end_time: Union[str, int, datetime, None] = None, 
                      limit: int = 1500) -> pd.DataFrame:
        """
        Fetch full history by chunking with retries.
        Supports flexible time parameters similar to fetch_binance_data.
        """
        if symbol in ['QQQUSDT', 'SPYUSDT', 'XAUUSDT', 'XAGUSDT']:
            start_ts = None
            end_ts = None
            max_days = get_yf_max_days(interval, years=years)
            if start_time is None and end_time is None:
                days = max_days
                start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                end_ts = int(datetime.now().timestamp() * 1000)
            else:
                if end_time is None:
                    end_ts = int(datetime.now().timestamp() * 1000)
                elif isinstance(end_time, (str, datetime)):
                    end_ts = int(pd.to_datetime(end_time).timestamp() * 1000)
                else:
                    end_ts = int(end_time)

                if isinstance(start_time, (str, datetime)):
                    start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
                elif start_time is not None:
                    start_ts = int(start_time)
                else:
                    days = max_days
                    start_ts = int((pd.to_datetime(end_ts, unit='ms') - timedelta(days=days)).timestamp() * 1000)
            return _fetch_yf_data(symbol, interval, start_ts=start_ts, end_ts=end_ts)
            
        # Calculate start_ts and end_ts in milliseconds
        if start_time is None and end_time is None:
            # use years if provided, else default to interval-appropriate lookback
            days = get_default_fetch_days(interval, years=years)
            start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            end_ts = int(datetime.now().timestamp() * 1000)
        else:
            # end_time defaults to now if not provided
            if end_time is None:
                end_ts = int(datetime.now().timestamp() * 1000)
            elif isinstance(end_time, (str, datetime)):
                end_ts = int(pd.to_datetime(end_time).timestamp() * 1000)
            else:
                end_ts = int(end_time)

            # start_time calculation
            if isinstance(start_time, (str, datetime)):
                start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
            elif start_time is not None:
                start_ts = int(start_time)
            else:
                # fall back to interval-appropriate lookback if start_time is None
                days = get_default_fetch_days(interval, years=years)
                start_ts = int((pd.to_datetime(end_ts, unit='ms') - timedelta(days=days)).timestamp() * 1000)

        all_dfs = []
        current_start = start_ts
        max_retries = 3
        
        while current_start < end_ts:
            df = pd.DataFrame()
            for attempt in range(max_retries):
                try:
                    df = self.fetch_klines(symbol, interval, start_time=current_start, end_time=end_ts, limit=limit)
                    if not df.empty:
                        break
                except requests.exceptions.HTTPError as e:
                    # If 418 (IP Banned) or 429 (Rate Limit), stop immediately
                    if e.response.status_code in [418, 429]:
                        logger.critical(f"Aborting fetch for {symbol} due to Rate Limit/Ban: {e}")
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {symbol} due to HTTP Error: {e}")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {symbol} due to: {e}")
                    time.sleep(1)
            
            if df.empty:
                break
                
            all_dfs.append(df)
            last_ts = int(df.iloc[-1]['open_time'].timestamp() * 1000)
            
            if last_ts >= end_ts or len(df) < limit:
                break
            
            current_start = last_ts + 1  # ensure forward progress
            time.sleep(self.rate_limit_delay)
            
        if not all_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_dfs).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        return final_df

    def fetch_candles(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch exactly N candles by chunking backwards from current time.
        """
        if symbol in ['QQQUSDT', 'SPYUSDT']:
            return _fetch_yf_data(symbol, interval, limit=limit)
            
        all_dfs = []
        current_end = int(datetime.now().timestamp() * 1000)
        remaining = limit
        
        while remaining > 0:
            fetch_limit = min(remaining, 1500)
            df = self.fetch_klines(symbol, interval, end_time=current_end, limit=fetch_limit)
            
            if df.empty:
                break
            
            all_dfs.append(df)
            remaining -= len(df)
            
            # Move current_end to the oldest candle's open time - 1ms
            current_end = int(df['open_time'].min().timestamp() * 1000) - 1
            
            if len(df) < fetch_limit:
                break
                
            time.sleep(self.rate_limit_delay)
            
        if not all_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_dfs).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        return final_df.tail(limit)


class DataManager:
    """Manages data storage and caching"""
    
    def __init__(self, data_dir: str = "data_cache", cache_size: int = 500):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.fetcher = BinanceFuturesFetcher()
        self._last_sync = {}
        self._universe_cache = None
        self._last_universe_fetch = 0
        
    def get_cached_symbols(self) -> List[str]:
        """Returns all valid USDT perpetual symbols found in local data_cache."""
        try:
            if not os.path.exists(self.data_dir):
                return []
            files = os.listdir(self.data_dir)
            symbols = set()
            for f in files:
                if f.endswith(".parquet") and "_" in f:
                    sym = f.split("_")[0]
                    if sym.endswith("USDT") and sym not in IGNORED_CRYPTO:
                        symbols.add(sym)
            return sorted(list(symbols))
        except Exception as e:
            logger.warning(f"Error reading cached symbols from disk: {e}")
            return []

    def get_all_symbols(self) -> List[str]:
        """Returns all available Binance USDT perpetual symbols (500+ symbols), merged with mandatory/cached symbols."""
        try:
            syms = self.fetcher.get_all_symbols()
            if syms:
                combined = set(syms)
                if MANDATORY_CRYPTO:
                    combined.update(MANDATORY_CRYPTO)
                combined.update(self.get_cached_symbols())
                return sorted(list(combined))
        except Exception as e:
            logger.error(f"Error fetching all symbols from Binance: {e}")

        # Fallback to cached symbols and mandatory crypto
        cached = set(self.get_cached_symbols())
        if MANDATORY_CRYPTO:
            cached.update(MANDATORY_CRYPTO)
        return sorted(list(cached))

    def get_universe(self, top_n: int = 1000) -> List[str]:
        """Returns top N symbols by volume, cached for 10 minutes session-wide."""
        now = time.time()
        if self._universe_cache and (now - self._last_universe_fetch < 600):
            return self._universe_cache[:top_n]
            
        syms = []
        try:
            # Fetch absolute all or up to 1000
            syms = self.fetcher.get_top_volume_symbols(top_n=1000)
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, 'status_code', 'Unknown')
            logger.error(f"HTTP Error fetching universe ({status_code}): {e}")
        except Exception as e:
            logger.error(f"Failed to fetch universe from Binance: {e}")
            
        if not syms:
            # Fallback 1: Use locally cached symbols on disk
            cached_syms = self.get_cached_symbols()
            if cached_syms:
                # Merge mandatory symbols with cached symbols
                syms = sorted(list(set(MANDATORY_CRYPTO).union(cached_syms)))
            else:
                # Fallback 2: Mandatory crypto list
                syms = list(MANDATORY_CRYPTO)
            
            # Temporary short cache (15s) so we retry network soon without pounding
            self._universe_cache = syms
            self._last_universe_fetch = now - 585
            return syms[:top_n]
            
        self._universe_cache = syms
        self._last_universe_fetch = now
        return syms[:top_n]
        
    def _get_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self.data_dir, f"{symbol}_{interval}.parquet")
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str):
        if data.empty: return
        path = self._get_path(symbol, interval)
        data.to_parquet(path, index=False)
        # Invalidate cache on save
        self.clear_cache(symbol, interval)
        
    def append_data(self, symbol: str, interval: str, new_data: pd.DataFrame):
        """Merges new data with existing cache, removes duplicates and saves."""
        if new_data.empty: return
        
        path = self._get_path(symbol, interval)
        if not os.path.exists(path):
            self.save_data(new_data, symbol, interval)
            return
            
        try:
            # Efficiently read just the max open_time
            existing_ts_df = pd.read_parquet(path, columns=['open_time'])
            if not existing_ts_df.empty:
                max_ts = existing_ts_df['open_time'].max()
                new_min_ts = new_data['open_time'].min()
                
                if pd.Timestamp(new_min_ts) > pd.Timestamp(max_ts):
                    # Fast path: strictly newer data, append directly
                    new_data.to_parquet(path, engine='fastparquet', append=True, index=False)
                    self.clear_cache(symbol, interval)
                    return
        except Exception as e:
            logger.warning(f"Fast append check failed for {symbol}_{interval}, falling back to full merge: {e}")
            
        existing = pd.read_parquet(path)
            
        # Merge and deduplicate
        combined = pd.concat([existing, new_data])
        combined = combined.drop_duplicates(subset=['open_time'], keep='last')
        combined = combined.sort_values('open_time').reset_index(drop=True)
        
        self.save_data(combined, symbol, interval)

    def get_cache_range(self, symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Returns (first_ts, last_ts) from the cached file without loading the full set."""
        path = self._get_path(symbol, interval)
        if not os.path.exists(path):
            return None, None
        try:
            # Efficiently read just the open_time column
            df = pd.read_parquet(path, columns=['open_time'])
            if df.empty: return None, None
            return df['open_time'].min(), df['open_time'].max()
        except Exception as e:
            logger.error(f"Error reading cache range for {symbol}_{interval}: {e}")
            return None, None

    def _sync_data(self, symbol: str, interval: str):
        """Automatically fetch and append missing data up to the current time."""
        if not interval or interval == 'None':
            # Skip if interval is missing or a string 'None'
            return

        cache_key = f"{symbol}_{interval}"
        now = time.time()
        
        path = self._get_path(symbol, interval)
        if hasattr(self, '_last_sync') and cache_key in self._last_sync:
            if os.path.exists(path) and (now - self._last_sync[cache_key] < 60):
                # Debounced
                return
                
        if not hasattr(self, '_last_sync'):
            self._last_sync = {}

        try:
            first_ts, last_ts = self.get_cache_range(symbol, interval)
            end_t = datetime.now()
            default_days = get_default_fetch_days(interval)
            if symbol in ['QQQUSDT', 'SPYUSDT', 'XAUUSDT', 'XAGUSDT']:
                default_days = min(default_days, get_yf_max_days(interval))
            
            if not first_ts:
                # No data: fetch full default history (respecting YF interval limits)
                days_to_fetch = default_days
                    
                logger.info(f"Auto-sync: Fetching initial {days_to_fetch} days for {symbol} ({interval})")
                requested_start = end_t - timedelta(days=days_to_fetch)
                df = self.fetcher.fetch_history(symbol, interval, start_time=requested_start, end_time=end_t)
                if not df.empty:
                    self.save_data(df, symbol, interval)
                    self._last_sync[cache_key] = now
            else:
                # 1. Forward sync: fetch new candles from last_ts
                start_ts_ms = int(pd.Timestamp(last_ts).tz_localize('UTC').timestamp() * 1000)
                df_fwd = self.fetcher.fetch_history(symbol, interval, start_time=start_ts_ms, end_time=end_t)
                if not df_fwd.empty and len(df_fwd) > 1:
                    logger.info(f"Auto-sync: Fetched {len(df_fwd)} new candles for {symbol} ({interval})")
                    self.append_data(symbol, interval, df_fwd)
                    self._last_sync[cache_key] = now

                # 2. Backward backfill: if cached data is shorter than required warmup lookback, backfill older candles
                required_start = end_t - timedelta(days=default_days)
                first_dt = pd.to_datetime(first_ts).tz_localize(None)
                if first_dt > required_start + timedelta(days=3):
                    logger.info(f"Auto-sync: Backfilling history for {symbol} ({interval}) from {required_start} to {first_dt}")
                    b_start_ms = int(pd.Timestamp(required_start).tz_localize('UTC').timestamp() * 1000)
                    b_end_ms = int(pd.Timestamp(first_dt).tz_localize('UTC').timestamp() * 1000)
                    df_back = self.fetcher.fetch_history(symbol, interval, start_time=b_start_ms, end_time=b_end_ms)
                    if not df_back.empty:
                        self.append_data(symbol, interval, df_back)
                        self._last_sync[cache_key] = now
        except Exception as e:
            logger.error(f"Error auto-syncing {symbol}_{interval}: {e}")

    def load_data(self, symbol: str, interval: str, auto_sync: bool = True, include_realtime: bool = False) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{interval}"
        path = self._get_path(symbol, interval)
        
        if auto_sync:
            self._sync_data(symbol, interval)
            
        # Check if file exists first
        if not os.path.exists(path):
            return None
            
        df = None
        # Check cache
        if cache_key in self._cache:
            # Check if file has been modified since cached
            mtime = os.path.getmtime(path)
            if self._cache[cache_key]['mtime'] >= mtime:
                df = self._cache[cache_key]['data']
        
        if df is None:
            # Load from disk
            try:
                df = pd.read_parquet(path)
                if not df.empty and 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time']) + timedelta(hours=TIMEZONE_OFFSET)
                    if 'close_time' in df.columns:
                        df['close_time'] = pd.to_datetime(df['close_time']) + timedelta(hours=TIMEZONE_OFFSET)
                # Simple LRU
                self._cache[cache_key] = {
                    'data': df,
                    'mtime': os.path.getmtime(path)
                }
                self._cache.move_to_end(cache_key)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                return None
        else:
            # Move accessed item to end
            self._cache.move_to_end(cache_key)

        # Append latest 1m data for realtime update
        if include_realtime and df is not None and not df.empty and interval != '1m':
            try:
                df_1m = self.fetcher.fetch_candles(symbol, '1m', limit=1)
                if not df_1m.empty:
                    df_1m['open_time'] = pd.to_datetime(df_1m['open_time']) + timedelta(hours=TIMEZONE_OFFSET)
                    if 'close_time' in df_1m.columns:
                        df_1m['close_time'] = pd.to_datetime(df_1m['close_time']) + timedelta(hours=TIMEZONE_OFFSET)
                    df = pd.concat([df, df_1m], ignore_index=True)
                    df = df.drop_duplicates(subset=['open_time'], keep='last').reset_index(drop=True)
            except Exception as e:
                logger.error(f"Error fetching 1m realtime data for {symbol}: {e}")
                
        return df

    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        if symbol and interval:
            cache_key = f"{symbol}_{interval}"
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()
            
    def delete_data(self, interval: str = "ALL"):
        """Deletes cached files for a specific interval or all data."""
        files = os.listdir(self.data_dir)
        deleted_count = 0
        for f in files:
            if not f.endswith(".parquet"):
                continue
            
            should_delete = False
            if interval == "ALL":
                should_delete = True
            else:
                # format: SYMBOL_INTERVAL.parquet
                name = f.replace(".parquet", "")
                parts = name.split('_')
                if len(parts) >= 2 and parts[-1] == interval:
                    should_delete = True
            
            if should_delete:
                try:
                    os.remove(os.path.join(self.data_dir, f))
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {f}: {e}")
        
        self.clear_cache()
        return deleted_count
        
    def get_existing_symbols(self) -> List[str]:
        # Legacy support or simple list
        return list(self.get_inventory().keys())

    def get_inventory(self) -> Dict[str, List[str]]:
        """Returns dictionary of {symbol: [interval, interval...]}"""
        files = os.listdir(self.data_dir)
        inventory = {}
        for f in files:
            if f.endswith(".parquet"):
                # format: SYMBOL_INTERVAL.parquet
                # Note: symbol might contain underscores? content usually doesn't.
                # safely split from right?
                # standard is SYMBOL_INTERVAL.parquet
                name = f.replace(".parquet", "")
                parts = name.split('_')
                if len(parts) >= 2:
                    interval = parts[-1]
                    symbol = "_".join(parts[:-1]) # join rest in case symbol has _
                    
                if symbol not in inventory:
                    inventory[symbol] = []
                inventory[symbol].append(interval)
        return inventory

    def get_cache_metadata(self) -> List[Dict]:
        """Returns detailed metadata for all cached files"""
        files = os.listdir(self.data_dir)
        metadata = []
        for f in files:
            if f.endswith(".parquet"):
                try:
                    path = os.path.join(self.data_dir, f)
                    # Parse name
                    name = f.replace(".parquet", "")
                    parts = name.split('_')
                    if len(parts) >= 2:
                        interval = parts[-1]
                        symbol = "_".join(parts[:-1])
                        
                        # Read minimal data (just open_time)
                        df = pd.read_parquet(path, columns=['open_time'])
                        if not df.empty:
                            start = df['open_time'].min()
                            end = df['open_time'].max()
                            count = len(df)
                            mtime = os.path.getmtime(path)
                            size_mb = os.path.getsize(path) / (1024 * 1024)
                            metadata.append({
                                'ticker': symbol,
                                'interval': interval,
                                'start_date': str(start),
                                'end_date': str(end),
                                'count': int(count),
                                'size_mb': float(size_mb),
                                'last_modified': float(mtime)
                            })
                except Exception as e:
                    logger.error(f"Error reading metadata for {f}: {e}")
        return metadata

def _fetch_yf_data(symbol: str, interval: str, start_ts: int = None, end_ts: int = None, limit: int = None) -> pd.DataFrame:
    yf_sym_map = {
        'XAUUSDT': 'GC=F',
        'XAGUSDT': 'SI=F',
        'QQQUSDT': 'QQQ',
        'SPYUSDT': 'SPY'
    }
    yf_sym = yf_sym_map.get(symbol, symbol.replace("USDT", ""))
    
    interval_map = {
        '1m': '1m', '3m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '1h', '4h': '1h', '6h': '1h', '8h': '1h', '12h': '1h',
        '1d': '1d', '3d': '1d', '1w': '1wk'
    }
    yf_interval = interval_map.get(interval, '1d')
    
    ticker = yf.Ticker(yf_sym)
    max_days = get_yf_max_days(interval)
    now_dt = datetime.now()
    min_allowed_dt = now_dt - timedelta(days=max_days)
    
    try:
        if start_ts and end_ts:
            start_date = pd.to_datetime(start_ts, unit='ms')
            end_date = pd.to_datetime(end_ts, unit='ms')
            if start_date < min_allowed_dt:
                start_date = min_allowed_dt
            if start_date >= end_date:
                start_date = end_date - timedelta(days=min(max_days, 1))
            df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
        elif limit:
            period_str = "max" if yf_interval in ['1d', '1wk'] else f"{max_days}d"
            df = ticker.history(period=period_str, interval=yf_interval)
            if not df.empty:
                df = df.tail(limit)
        else:
            period_str = "max" if yf_interval in ['1d', '1wk'] else f"{max_days}d"
            df = ticker.history(period=period_str, interval=yf_interval)
    except Exception as e:
        logger.error(f"Failed to fetch YF data for {yf_sym}: {e}")
        return pd.DataFrame()
        
    if df.empty:
        return pd.DataFrame()
        
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={
        date_col: 'open_time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Check if tz-aware
    if df['open_time'].dt.tz is not None:
        df['open_time'] = df['open_time'].dt.tz_convert('UTC').dt.tz_localize(None)
        
    # Ensure uniform nanosecond timestamps matching Binance parquet dtypes
    df['open_time'] = pd.to_datetime(df['open_time']).astype('datetime64[ns]')

    # For hourly candles, floor to the hour so US equity (:30) candles align with Binance (:00) hourly candles
    if interval_map.get(interval, '1d') in ['1h', '2h', '4h', '6h', '8h', '12h']:
        df['open_time'] = df['open_time'].dt.floor('h')

    # Normalize to midnight to align with Binance daily candles
    if interval_map.get(interval, '1d') in ['1d', '1wk']:
        df['open_time'] = df['open_time'].dt.normalize()
        
    df = df.drop_duplicates(subset=['open_time'], keep='last')
        
    df['quote_volume'] = df['close'] * df['volume']
    
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
