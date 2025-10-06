# -*- coding: utf-8 -*-
"""
MIME-B — RF Futures Bot (BingX Perp, CCXT)
- Entries: TradingView Range Filter EXACT (BUY/SELL) — same bar close
- Size: 60% balance × leverage (default 10x)
- Exit/Management (after entry only):
  • Opposite RF signal ALWAYS closes
  • Smart Profit: TP1 partial + move to breakeven + ATR trailing (trend-riding)
  • Hold-TP in strong/up-strengthening trend (ADX/DI + candles)
  • Scale-In (incremental adds) while trend strengthens (bounded & cooled)
- Indicators: RSI/DI+/DI-/DX/ADX/ATR + Candlestick patterns
- HUD: colored, icons, status LED, candle-close countdown, WHY no-trade
- Robust keepalive (/ /metrics) — unchanged

NOTE:
- I did NOT change core function names. I only added helpers and extended logic.
"""

import os, time, math, threading, requests, traceback, random
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime, timedelta

# ------------ console colors ------------
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ------------ ENV ------------
def getenv(k, d=None, typ=str):
    v = os.getenv(k, d)
    if v is None: return d
    if typ is bool:  return str(v).lower() in ("1","true","yes","y","on")
    if typ is int:   return int(float(v))
    if typ is float: return float(v)
    return v

API_KEY    = getenv("BINGX_API_KEY", "")
API_SECRET = getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SYMBOL     = getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = getenv("INTERVAL", "15m")
LEVERAGE   = getenv("LEVERAGE", 10, int)
RISK_ALLOC = getenv("RISK_ALLOC", 0.60, float)

# Range Filter params (EXACT as TV script)
RF_SOURCE  = getenv("RF_SOURCE", "close").lower()
RF_PERIOD  = getenv("RF_PERIOD", 20, int)
RF_MULT    = getenv("RF_MULT", 3.5, float)
USE_TV_BAR = getenv("USE_TV_BAR", False, bool)     # False => last CLOSED bar (safer, TV-like)

# Indicators
RSI_LEN = getenv("RSI_LEN", 14, int)
ADX_LEN = getenv("ADX_LEN", 14, int)
ATR_LEN = getenv("ATR_LEN", 14, int)

# Execution guards
SPREAD_GUARD_BPS = getenv("SPREAD_GUARD_BPS", 6, int)
COOLDOWN_AFTER_CLOSE_BARS = getenv("COOLDOWN_AFTER_CLOSE_BARS", 0, int)

# Strategy mode
STRATEGY = getenv("STRATEGY", "smart").lower()      # smart | pure
USE_SMART_EXIT = getenv("USE_SMART_EXIT", True, bool)

# Smart Profit params
TP1_PCT          = getenv("TP1_PCT", 0.40, float)        # partial at +0.40%
TP1_CLOSE_FRAC   = getenv("TP1_CLOSE_FRAC", 0.50, float) # close 50% at TP1
BREAKEVEN_AFTER  = getenv("BREAKEVEN_AFTER_PCT", 0.30, float) # BE after +0.30%
TRAIL_ACTIVATE   = getenv("TRAIL_ACTIVATE_PCT", 0.60, float)  # start trailing after +0.60%
ATR_MULT_TRAIL   = getenv("ATR_MULT_TRAIL", 1.6, float)       # trail distance

# Smart Scale/Hold params (NEW — post-entry only)
SCALE_IN               = getenv("SCALE_IN", True, bool)
SCALE_MAX_ADDS         = getenv("SCALE_MAX_ADDS", 2, int)
SCALE_ADD_FRAC         = getenv("SCALE_ADD_FRAC", 0.25, float)        # add 25% of current qty
SCALE_MIN_RR_FOR_ADD   = getenv("SCALE_MIN_RR_FOR_ADD", 0.30, float)  # profit % to allow first add
SCALE_ADX_MIN          = getenv("SCALE_ADX_MIN", 25, int)
SCALE_ADX_SLOPE_MIN    = getenv("SCALE_ADX_SLOPE_MIN", 0.50, float)   # ADX rising
SCALE_COOLDOWN_BARS    = getenv("SCALE_COOLDOWN_BARS", 3, int)

HOLD_TP_STRONG         = getenv("HOLD_TP_STRONG", True, bool)
HOLD_TP_ADX            = getenv("HOLD_TP_ADX", 28, int)
HOLD_TP_SLOPE          = getenv("HOLD_TP_SLOPE", 0.50, float)

# pacing / keepalive
SLEEP_S  = getenv("DECISION_EVERY_S", 30, int)
SELF_URL = getenv("SELF_URL", "") or getenv("RENDER_EXTERNAL_URL","")
KEEPALIVE_SECONDS = getenv("KEEPALIVE_SECONDS", 50, int)
PORT     = getenv("PORT", 5000, int)

FORCE_TV_ENTRIES = getenv("FORCE_TV_ENTRIES", True, bool)  # تأكيد أن الدخول من RF فقط
print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"KEEPALIVE: url={'SET' if SELF_URL else 'NOT SET'} • every {KEEPALIVE_SECONDS}s", "yellow"))

# ------------ Exchange ------------
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_exchange()
try:
    ex.load_markets()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# ------------ Helpers ------------
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: return fn()
        except Exception:
            if i == attempts-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        print(colored(f"❌ ticker: {e}", "red")); return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception as e:
        print(colored(f"❌ balance: {e}", "red")); return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def secs_to_candle_close():
    try:
        if INTERVAL.endswith("m"):
            m = int(INTERVAL[:-1])
        elif INTERVAL.endswith("h"):
            m = int(INTERVAL[:-1]) * 60
        else:
            m = 15
        now = datetime.utcnow()
        total = m*60
        sec_into = (now.minute % m)*60 + now.second
        return max(0, total - sec_into)
    except Exception:
        return None

# ------------ Indicators ------------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1+rs))

    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# ------------ Candles (patterns) ------------
def _candle(df, i):
    return {
        "o": float(df["open"].iloc[i]),
        "h": float(df["high"].iloc[i]),
        "l": float(df["low"].iloc[i]),
        "c": float(df["close"].iloc[i])
    }

def detect_candle_pattern(df: pd.DataFrame):
    """Lightweight detector for key patterns (on the same bar index used for indicators)."""
    if len(df) < 3: return "NONE"
    i = len(df)-1 if USE_TV_BAR else len(df)-2
    a = _candle(df, i-2); b = _candle(df, i-1); c = _candle(df, i)
    def body(x): return abs(x["c"]-x["o"])
    def upwick(x): return x["h"] - max(x["o"],x["c"])
    def lowick(x): return min(x["o"],x["c"]) - x["l"]
    def bull(x): return x["c"]>x["o"]
    def bear(x): return x["o"]>x["c"]
    avg_body = (body(a)+body(b)+body(c))/3 or 1e-9

    # Primary patterns we use in management logic
    if bull(c) and bear(b) and (c["o"]<=b["c"]) and (c["c"]>=b["o"]): return "ENGULF_BULL"
    if bear(c) and bull(b) and (c["o"]>=b["c"]) and (c["c"]<=b["o"]): return "ENGULF_BEAR"

    if bull(c) and lowick(c) >= 2*body(c) and upwick(c) <= body(c)*0.5: return "HAMMER"
    if bear(c) and upwick(c) >= 2*body(c) and lowick(c) <= body(c)*0.5: return "SHOOTING_STAR"

    if bear(a) and (body(b)<=0.25*avg_body) and bull(c) and c["c"] > (a["o"]+a["c"])/2: return "MORNING_STAR"
    if bull(a) and (body(b)<=0.25*avg_body) and bear(c) and c["c"] < (a["o"]+a["c"])/2: return "EVENING_STAR"

    if (body(c)<=0.25*avg_body) and (upwick(c)+lowick(c) > body(c)*2): return "DOJI"

    return "NONE"

# ------------ Range Filter (EXACT) ------------
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def compute_tv_signals(df: pd.DataFrame):
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward = (fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f&((src_gt_p)|(src_lt_p))&(upward>0))
    shortCond=(src_lt_f&((src_lt_p)|(src_gt_p))&(downward>0))
    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal=longCond&(CondIni.shift(1)==-1)
    shortSignal=shortCond&(CondIni.shift(1)==1)
    i=len(df)-1 if USE_TV_BAR else len(df)-2
    return {
        "time": int(df["time"].iloc[i]), "price": float(df["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

# ------------ State & Sync ------------
state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
       "adds_done":0,"last_add_bar":-10,"hold_tp":False}
compound_pnl=0.0
last_signal_id=None
post_close_cooldown=0
_last_df=None  # global ref for smart logic (no signature changes)

def compute_size(balance, price):
    capital = balance*RISK_ALLOC*LEVERAGE
    return max(0.0, capital/max(price,1e-9))

def sync_from_exchange_once():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = p.get("symbol") or p.get("info",{}).get("symbol") or ""
            if SYMBOL.split(":")[0] not in sym: 
                continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: continue
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            if side not in ("long","short"):
                cost = float(p.get("cost") or 0.0)
                side = "long" if cost>0 else "short"
            state.update({"open":True,"side":side,"entry":entry,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
                          "adds_done":0,"last_add_bar":-10,"hold_tp":False})
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))

# ------------ Orders ------------
def open_market(side, qty, price):
    global state
    if qty<=0: 
        print(colored("❌ qty<=0 skip open","red")); return
    if MODE_LIVE:
        try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e: print(colored(f"⚠️ set_leverage: {e}","yellow"))
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":False})
        except Exception as e: print(colored(f"❌ open: {e}","red"))
    state={"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
           "adds_done":0,"last_add_bar":-10,"hold_tp":False}
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))

def add_to_position(frac, price):
    """Scale-In add — new helper (does NOT change original funcs)."""
    global state
    if not state["open"] or frac<=0: return
    add_qty = max(0.0, state["qty"]*frac)
    if add_qty<=0: return
    side = "buy" if state["side"]=="long" else "sell"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,add_qty,params={"reduceOnly":False})
        except Exception as e:
            print(colored(f"❌ scale-in order: {e}","red")); return
    old_qty=state["qty"]; old_entry=state["entry"]
    new_entry=((old_entry*old_qty)+(price*add_qty))/(old_qty+add_qty)
    state["qty"]=old_qty+add_qty; state["entry"]=new_entry
    print(colored(f"➕ SCALE-IN add={fmt(add_qty,4)} @ {fmt(price)} new_qty={fmt(state['qty'],4)} new_entry={fmt(state['entry'])}","blue"))

def close_partial(frac, reason):
    """Close fraction of current position (smart TP1)."""
    global state, compound_pnl
    if not state["open"]: return
    qty_close = max(0.0, state["qty"]*min(max(frac,0.0),1.0))
    if qty_close<=0: return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ partial close: {e}","red"))
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    if state["qty"]<=0:
        reset_after_full_close("fully_exited")

def reset_after_full_close(reason):
    global state, post_close_cooldown
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
           "adds_done":0,"last_add_bar":-10,"hold_tp":False}
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ close: {e}","red"))
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    reset_after_full_close(reason)

# ------------ Smart Profit & Intelligence ------------
def trend_strength_info(df: pd.DataFrame):
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ADX_LEN)
    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move>down_move) & (up_move>0), 0.0)
    minus_dm = down_move.where((down_move>up_move) & (down_move>0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)
    i = len(df)-1 if USE_TV_BAR else len(df)-2
    if len(adx)<6: slope=0.0
    else: slope=float(adx.iloc[i] - adx.iloc[i-4:i].mean())
    return {
        "adx": float(adx.iloc[i]), "slope": float(slope),
        "plus_di": float(plus_di.iloc[i]), "minus_di": float(minus_di.iloc[i])
    }

def smart_exit_check(info, ind):
    """Return True if full close happened. (POST-ENTRY ONLY management)"""
    global state, _last_df
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]):
        return None

    px = info["price"]; e=state["entry"]; side=state["side"]
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0

    # انتظر كام شمعة بعد الدخول لتجنب الخروج السريع
    if state["bars"] < 2:
        return None

    # --- Hold-TP logic (لا جني مبكر لو الترند قوي وبيتقوى) ---
    try:
        tinfo = trend_strength_info(_last_df) if _last_df is not None else {"adx":ind.get("adx",0.0),"slope":0.0,"plus_di":ind.get("plus_di",0.0),"minus_di":ind.get("minus_di",0.0)}
        strong = (HOLD_TP_STRONG and (tinfo["adx"]>=HOLD_TP_ADX) and (tinfo["slope"]>=HOLD_TP_SLOPE))
        if side=="long":
            state["hold_tp"] = strong and (tinfo["plus_di"]>tinfo["minus_di"])
        else:
            state["hold_tp"] = strong and (tinfo["minus_di"]>tinfo["plus_di"])
    except Exception:
        state["hold_tp"]=False

    # TP1 جزئي (لو مش عامل Hold-TP)
    if (not state["tp1_done"]) and (not state.get("hold_tp")) and rr >= TP1_PCT:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
        state["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER:
            state["breakeven"]=e

    # تحريك التريل بعد TP1
    if rr >= TRAIL_ACTIVATE and atr and ATR_MULT_TRAIL>0:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]:
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
                return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
                return True
    return None

def smart_scale_in(info, ind):
    """Scale-in adds while trend strengthens — POST-ENTRY ONLY."""
    global state, _last_df
    if not (SCALE_IN and state["open"]): return
    # ربح نسبي %
    px = info["price"]; e = state["entry"]
    rr = (px - e)/e * 100.0 * (1 if state["side"]=="long" else -1)
    if rr < SCALE_MIN_RR_FOR_ADD: return

    try:
        tinfo = trend_strength_info(_last_df) if _last_df is not None else {"adx":ind.get("adx",0.0),"slope":0.0,"plus_di":ind.get("plus_di",0.0),"minus_di":ind.get("minus_di",0.0)}
    except Exception:
        return
    adx = tinfo["adx"]; slope=tinfo["slope"]; plus_di=tinfo["plus_di"]; minus_di=tinfo["minus_di"]

    if not (adx >= SCALE_ADX_MIN and slope >= SCALE_ADX_SLOPE_MIN):
        return
    if state["side"]=="long" and not (plus_di > minus_di): return
    if state["side"]=="short" and not (minus_di > plus_di): return

    # تبريد وعدد مرات
    if state["adds_done"] >= SCALE_MAX_ADDS: return
    if state["bars"] - state["last_add_bar"] < SCALE_COOLDOWN_BARS: return

    add_to_position(SCALE_ADD_FRAC, px)
    state["adds_done"] += 1
    state["last_add_bar"] = state["bars"]

# ------------ HUD ------------
def snapshot(bal,info,ind,spread_bps,reason=None, pattern="NONE"):
    led = "🟩" if state["open"] and state["side"]=="long" else ("🟥" if state["open"] and state["side"]=="short" else "🟨")
    print(colored("─"*110,"cyan"))
    print(colored(f"{led} {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    # Indicators
    print("📈 INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))}  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   RSI({RSI_LEN})={fmt(ind['rsi'])}  +DI={fmt(ind['plus_di'])}  -DI={fmt(ind['minus_di'])}  DX={fmt(ind['dx'])}  ADX({ADX_LEN})={fmt(ind['adx'])}  ATR={fmt(ind['atr'])}")
    print(f"   🕯️ Candle={pattern}   ✅ BUY={info['long']}   ❌ SELL={info['short']}   🧮 spread_bps={fmt(spread_bps,2)}   Mode={STRATEGY.upper()}")
    # Position
    print()
    print("🧭 POSITION")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        mode = "HOLD-TP" if state.get("hold_tp") else ("SCALING" if state.get("adds_done",0)>0 else "NORMAL")
        print(f"   📌 {'🟩 LONG' if state['side']=='long' else '🟥 SHORT'}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}  Trail={fmt(state['trail'])}  TP1_done={state['tp1_done']}  Adds={state.get('adds_done',0)}  Mode={mode}")
    else:
        print("   ⚪ FLAT")
    # Results
    print()
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        secs = secs_to_candle_close()
        eta = f" • ⏱️ close in ~{secs}s" if secs is not None else ""
        print(colored(f"   ℹ️ No trade — reason: {reason}{eta}","yellow"))
    print(colored("─"*110,"cyan"))

# ------------ Decision Loop ------------
def trade_loop():
    global last_signal_id, state, post_close_cooldown, _last_df
    sync_from_exchange_once()
    while True:
        try:
            bal=balance_usdt()
            px=price_now()
            df=fetch_ohlcv()
            _last_df = df.copy()  # keep latest for smart logic
            info=compute_tv_signals(df)
            ind=compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            pattern = detect_candle_pattern(df)

            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # Smart management (AFTER ENTRY ONLY)
            smart_exit_check(info, ind)
            smart_scale_in(info, ind)

            # Decide entry — RF only (TV-like)
            sig="buy" if info["long"] else ("sell" if info["short"] else None)
            entry_type="RF" if sig else None
            reason=None

            # Guards (do NOT block RF logic except execution hygiene)
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps>SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown>0:
                reason=f"cooldown {post_close_cooldown} bars"

            # Close on opposite RF signal ALWAYS
            if state["open"] and sig and (reason is None):
                desired="long" if sig=="buy" else "short"
                if state["side"]!=desired:
                    close_market("opposite_signal")
                    qty=compute_size(bal, px)
                    if qty>0:
                        open_market(sig, qty, px)
                        last_signal_id=f"{info['time']}:{sig}"
                        snapshot(bal,info,ind,spread_bps,None,pattern)
                        time.sleep(SLEEP_S); continue

            # Open new position when flat (exactly RF)
            if not state["open"] and (reason is None) and sig:
                qty=compute_size(bal, px)
                if qty>0:
                    open_market(sig, qty, px)
                    last_signal_id=f"{info['time']}:{sig}"
                else:
                    reason="qty<=0"

            snapshot(bal,info,ind,spread_bps,reason,pattern)

            if state["open"]:
                state["bars"] += 1
            if post_close_cooldown>0 and not state["open"]:
                post_close_cooldown -= 1

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
        time.sleep(SLEEP_S)

# ------------ Keepalive + API ------------
def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL/RENDER_EXTERNAL_URL not set — skipping.", "yellow"))
        return
    sess = requests.Session()
    sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"🛰️ keepalive: ping {url} every {KEEPALIVE_SECONDS}s","cyan"))
    while True:
        try:
            r = sess.get(url, timeout=8)
            if r.status_code==200:
                print(colored("🟢 keepalive ok (200)","green"))
            else:
                print(colored(f"🟠 keepalive status={r.status_code}","yellow"))
        except Exception as e:
            print(colored(f"🔴 keepalive error: {e}","red"))
        time.sleep(max(KEEPALIVE_SECONDS,15))

app = Flask(__name__)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ MIME-B RF Bot — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategy": STRATEGY
    })

@app.route("/ping")
def ping(): return "pong", 200

# Boot
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
