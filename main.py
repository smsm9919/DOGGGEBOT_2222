import os, time, math, threading
from datetime import datetime, timezone
from termcolor import colored
from flask import Flask, jsonify
import pandas as pd

try:
    import ccxt
except Exception:
    ccxt = None

app = Flask(__name__)

ENV = lambda k, d=None: os.getenv(k, d)
SYMBOL = ENV("SYMBOL", "DOGE/USDT:USDT")
INTERVAL = ENV("INTERVAL", "15m")
DECISION_EVERY_S = int(ENV("DECISION_EVERY_S", "30"))
KEEPALIVE_SECONDS = int(ENV("KEEPALIVE_SECONDS", "60"))
PORT = int(ENV("PORT", "5000"))
LEVERAGE = int(ENV("LEVERAGE", "10"))
LIVE_TRADING = ENV("LIVE_TRADING", "false").lower() == "true"

STRATEGY = ENV("STRATEGY", "smart")
USE_TV_BAR = ENV("USE_TV_BAR", "false").lower() == "true"
FORCE_TV_ENTRIES = ENV("FORCE_TV_ENTRIES", "false").lower() == "true"

RF_PERIOD = int(ENV("RF_PERIOD", "20"))
RF_SOURCE = ENV("RF_SOURCE", "close")
RF_MULT = float(ENV("RF_MULT", "3.5"))
SPREAD_GUARD_BPS = int(ENV("SPREAD_GUARD_BPS", "6"))

RSI_LEN = int(ENV("RSI_LEN", "14"))
ADX_LEN = int(ENV("ADX_LEN", "14"))

TP1_PCT = float(ENV("TP1_PCT", "0.40"))
TP1_CLOSE_FRAC = float(ENV("TP1_CLOSE_FRAC", "0.50"))
TRAIL_ACTIVATE_PCT = float(ENV("TRAIL_ACTIVATE_PCT", "0.60"))
ATR_LEN = int(ENV("ATR_LEN", "14"))
ATR_MULT_TRAIL = float(ENV("ATR_MULT_TRAIL", "1.6"))
BREAKEVEN_AFTER_PCT = float(ENV("BREAKEVEN_AFTER_PCT", "0.30"))
COOLDOWN_AFTER_CLOSE_BARS = int(ENV("COOLDOWN_AFTER_CLOSE_BARS", "0"))
USE_SMART_EXIT = ENV("USE_SMART_EXIT", "true").lower() == "true"

SCALE_IN_ENABLED = ENV("SCALE_IN_ENABLED", "true").lower() == "true"
SCALE_IN_ADX_MIN = int(ENV("SCALE_IN_ADX_MIN", "25"))
SCALE_IN_SLOPE_MIN = float(ENV("SCALE_IN_SLOPE_MIN", "0.50"))
SCALE_IN_MAX_ADDS = int(ENV("SCALE_IN_MAX_ADDS", "3"))

RENDER_EXTERNAL_URL = ENV("RENDER_EXTERNAL_URL", "")

exchange = None
if LIVE_TRADING and ccxt is not None:
    exchange = ccxt.bingx({
        "apiKey": ENV("BINGX_API_KEY", ""),
        "secret": ENV("BINGX_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df, length=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = atr(df, 1)
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / (tr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / (tr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    adx_val = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx_val, plus_di, minus_di, dx

def range_filter(series, period=20, mult=3.5):
    ema = series.ewm(span=period, adjust=False).mean()
    dev = (series - ema).abs().ewm(span=period, adjust=False).mean()
    upper = ema + mult * dev
    lower = ema - mult * dev
    buy = (series.shift(1) <= upper.shift(1)) & (series > upper)
    sell = (series.shift(1) >= lower.shift(1)) & (series < lower)
    return ema, upper, lower, buy.astype(int), sell.astype(int)

def detect_candle(df):
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, 1e-9)
    upper_wick = df["high"] - df[["open","close"]].max(axis=1)
    lower_wick = df[["open","close"]].min(axis=1) - df["low"]
    doji = (body / range_) < 0.1
    pin_bull = (lower_wick / range_) > 0.6
    pin_bear = (upper_wick / range_) > 0.6
    prev = df.shift(1)
    bull_engulf = (df["close"] > df["open"]) & (prev["close"] < prev["open"]) & (df["close"] >= prev["open"]) & (df["open"] <= prev["close"])
    bear_engulf = (df["close"] < df["open"]) & (prev["close"] > prev["open"]) & (df["close"] <= prev["open"]) & (df["open"] >= prev["close"])
    return doji, pin_bull, pin_bear, bull_engulf, bear_engulf

def fetch_ohlcv(symbol, timeframe="15m", limit=500):
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex = exchange if exchange else ccxt.bingx({"enableRateLimit": True, "options":{"defaultType":"swap"}})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

state = {"position": None, "cooldown": 0, "last_keepalive": 0, "last_decision": 0}

def log_line(icon, msg, color=None):
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    s = f"{dt}  {icon}  {msg}"
    print(colored(s, color) if color else s, flush=True)

def decide():
    try:
        df = fetch_ohlcv(SYMBOL, INTERVAL, limit=500)
    except Exception as e:
        log_line("🛑", f"fetch error: {e}", "red")
        return

    df = df.set_index("ts")
    ema, up, lo, buy_sig, sell_sig = range_filter(df[RF_SOURCE], RF_PERIOD, RF_MULT)
    df["ema"] = ema; df["up"] = up; df["lo"] = lo
    df["buy_sig"] = buy_sig; df["sell_sig"] = sell_sig
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    df["adx"], df["+di"], df["-di"], df["dx"] = adx(df, ADX_LEN)
    doji, pin_bull, pin_bear, bull_engulf, bear_engulf = detect_candle(df)
    df["doji"]=doji; df["pin_bull"]=pin_bull; df["pin_bear"]=pin_bear
    df["bull_engulf"]=bull_engulf; df["bear_engulf"]=bear_engulf

    last = df.iloc[-1]
    price = float(last["close"])
    pos = state["position"]

    regime_up = last["adx"] > 25 and last["+di"] > last["-di"]
    regime_dn = last["adx"] > 25 and last["-di"] > last["+di"]

    long_ready = (last["buy_sig"]==1) and (bull_engulf.iloc[-1] or last["pin_bull"] or (last["rsi"]>45))
    short_ready= (last["sell_sig"]==1) and (bear_engulf.iloc[-1] or last["pin_bear"] or (last["rsi"]<55))

    allow_long = long_ready and (regime_up or FORCE_TV_ENTRIES)
    allow_short= short_ready and (regime_dn or FORCE_TV_ENTRIES)

    if pos:
        side = pos["side"]
        qty = pos["qty"]; entry = pos["entry"]
        uPnL = (price-entry) * (1 if side=="long" else -1)
        pnl_pct = uPnL / entry

        if not pos["tp1_done"] and pnl_pct >= TP1_PCT/100.0:
            close_qty = qty * TP1_CLOSE_FRAC
            pos["qty"] -= close_qty
            pos["tp1_done"] = True
            pos["breakeven"] = entry
            log_line("🏁", f"TP1 close {close_qty:.4f} @ ~{price:.6f} | protect BE", "green")

        if pos.get("breakeven") and pnl_pct >= BREAKEVEN_AFTER_PCT/100.0:
            pos["breakeven"] = entry

        if pnl_pct >= TRAIL_ACTIVATE_PCT/100.0:
            trail = price - (ATR_MULT_TRAIL*last["atr"]) if side=="long" else price + (ATR_MULT_TRAIL*last["atr"])
            pos["trail"] = max(pos.get("trail", -1e9), trail) if side=="long" else min(pos.get("trail", 1e9), trail)
            if (side=="long" and price < pos["trail"]) or (side=="short" and price > pos["trail"]):
                log_line("🧹", f"Trail exit {side} @ {price:.6f}", "yellow")
                state["position"] = None
                state["cooldown"] = COOLDOWN_AFTER_CLOSE_BARS
                return

        if USE_SMART_EXIT and last["adx"] < 18 and pos["tp1_done"]:
            log_line("⚡", f"Smart exit (ranging) {side} @ {price:.6f}", "yellow")
            state["position"] = None
            state["cooldown"] = COOLDOWN_AFTER_CLOSE_BARS
            return

        if SCALE_IN_ENABLED and pos["adds"] < SCALE_IN_MAX_ADDS:
            slope = (df["ema"].iloc[-1] - df["ema"].iloc[-4]) / (3 + 1e-9)
            strong = (last["adx"] >= SCALE_IN_ADX_MIN) and ((slope> SCALE_IN_SLOPE_MIN and side=="long") or (slope< -SCALE_IN_SLOPE_MIN and side=="short"))
            if strong:
                add_qty = qty * 0.25
                pos["qty"] += add_qty
                pos["adds"] += 1
                log_line("➕", f"Scale-in add {add_qty:.4f} side={side} (ADX={last['adx']:.2f}, slope={slope:.3f})", "cyan")
        return

    if state["cooldown"] > 0:
        state["cooldown"] -= 1
        log_line("⏳", f"Cooldown… bars left {state['cooldown']}", "yellow")
        return

    if allow_long and not allow_short:
        qty = 100.0 / price
        state["position"] = {"side":"long","qty":qty,"entry":price,"adds":0,"tp1_done":False,"trail":None}
        log_line("🟢", f"LONG entry {qty:.4f} @ {price:.6f} | ADX={last['adx']:.2f} RSI={last['rsi']:.1f}", "green")
        return
    if allow_short and not allow_long:
        qty = 100.0 / price
        state["position"] = {"side":"short","qty":qty,"entry":price,"adds":0,"tp1_done":False,"trail":None}
        log_line("🔴", f"SHORT entry {qty:.4f} @ {price:.6f} | ADX={last['adx']:.2f} RSI={last['rsi']:.1f}", "red")
        return

    log_line("🔵", "No trade – reason: no signal", "blue")

def worker():
    while True:
        now = time.time()
        if now - state["last_decision"] >= DECISION_EVERY_S:
            state["last_decision"] = now
            decide()
        if now - state["last_keepalive"] >= KEEPALIVE_SECONDS:
            state["last_keepalive"] = now
            log_line("💙", "keepalive ok (200)")
        time.sleep(1)

@app.route("/")
def home():
    return "dogee-bot-z ok"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "strategy": STRATEGY,
        "position": state["position"],
        "cooldown": state["cooldown"],
        "interval": INTERVAL,
        "leverage": LEVERAGE,
        "live_trading": LIVE_TRADING,
        "url": RENDER_EXTERNAL_URL,
    })

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
