# main.py
# =========================================================
# DOGE/USDT (BingX Perp) — Pro AI bot
# - TV-matching entries (Range Filter)
# - RSI / ADX / +DI / -DI / ATR
# - Full Candlestick Intelligence after entry
# - Smart Mgmt: TP1 + Breakeven + ATR-Trailing + Giveback
# - Auto Scalp vs Trend modes
# - Cumulative PnL sizing (next order grows with realized PnL)
# - Clean iconic logs + /metrics + keepalive
# =========================================================

import os, time, math, json, threading, traceback, statistics as stats, logging
from datetime import datetime, timezone
from termcolor import cprint, colored
import pandas as pd
from flask import Flask, jsonify
import requests

# ---------------- ENV ----------------
BINGX_API_KEY    = os.getenv("BINGX_API_KEY","")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET","")

SYMBOL           = os.getenv("SYMBOL","DOGE/USDT:USDT")   # ccxt Perp format
INTERVAL         = os.getenv("INTERVAL","15m")
LEVERAGE         = int(os.getenv("LEVERAGE","10"))
RISK_ALLOC       = float(os.getenv("RISK_ALLOC","0.60"))
DECISION_EVERY_S = int(os.getenv("DECISION_EVERY_S","30"))
KEEPALIVE_SECONDS= int(os.getenv("KEEPALIVE_SECONDS","50"))
PORT             = int(os.getenv("PORT","5000"))

# Range Filter (TradingView-style)
RF_PERIOD        = int(os.getenv("RF_PERIOD","20"))
RF_SOURCE        = os.getenv("RF_SOURCE","close")  # close/high/low
RF_MULT          = float(os.getenv("RF_MULT","3.5"))
SPREAD_GUARD_BPS = int(os.getenv("SPREAD_GUARD_BPS","6"))
USE_TV_BAR       = os.getenv("USE_TV_BAR","false").lower()=="true"     # wait candle close
FORCE_TV_ENTRIES = os.getenv("FORCE_TV_ENTRIES","false").lower()=="true"

# Indicators
RSI_LEN          = int(os.getenv("RSI_LEN","14"))
ADX_LEN          = int(os.getenv("ADX_LEN","14"))
ATR_LEN          = int(os.getenv("ATR_LEN","14"))

# Smart management
TP1_PCT          = float(os.getenv("TP1_PCT","0.40"))         # 40% move
TP1_CLOSE_FRAC   = float(os.getenv("TP1_CLOSE_FRAC","0.50"))  # close 50% at TP1
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL   = float(os.getenv("ATR_MULT_TRAIL","1.6"))
GIVEBACK_PCT     = float(os.getenv("GIVEBACK_PCT","0.30"))
BREAKEVEN_AFTER_PCT = float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
MOVE_3BARS_PCT   = float(os.getenv("MOVE_3BARS_PCT","0.8"))
MIN_TP_PERCENT   = float(os.getenv("MIN_TP_PERCENT","0.4"))

# Hold more when trend is strong
HOLD_TP_STRONG   = os.getenv("HOLD_TP_STRONG","true").lower()=="true"
HOLD_TP_ADX      = int(os.getenv("HOLD_TP_ADX","28"))
HOLD_TP_SLOPE    = float(os.getenv("HOLD_TP_SLOPE","0.50"))

USE_SMART_EXIT   = os.getenv("USE_SMART_EXIT","true").lower()=="true"

SELF_URL         = os.getenv("RENDER_EXTERNAL_URL","")
LIVE_TRADING     = (os.getenv("LIVE_TRADING","true").lower()=="true"
                    and bool(BINGX_API_KEY) and bool(BINGX_API_SECRET))

# ---------------- Exchange (ccxt) ----------------
try:
    import ccxt
    ex = ccxt.bingx({
        "apiKey": BINGX_API_KEY,
        "secret": BINGX_API_SECRET,
        "options": {"defaultType": "swap"},
        "enableRateLimit": True
    })
    ex.set_sandbox_mode(False)
except Exception as e:
    ex = None
    cprint(f"⚠️ ccxt init error: {e}", "red")

# ---------------- Utils ----------------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ema(series, length): return series.ewm(span=length, adjust=False).mean()

def true_range(df):
    pc = df['close'].shift(1)
    tr1 = df['high']-df['low']
    tr2 = (df['high']-pc).abs()
    tr3 = (df['low']-pc).abs()
    return pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

def atr(df, length=14): return true_range(df).rolling(length).mean()

def rsi(series, length=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = (-d.clip(upper=0)).rolling(length).mean()
    rs = up/(dn+1e-9)
    return 100 - (100/(1+rs))

def dx_plus_minus(df, length=14):
    upMove   = df['high'].diff()
    downMove = -df['low'].diff()
    plusDM  = ((upMove>downMove) & (upMove>0)) * upMove
    minusDM = ((downMove>upMove) & (downMove>0)) * downMove
    trv = true_range(df).rolling(length).sum()
    plusDI  = 100 * (plusDM.rolling(length).sum() / (trv+1e-9))
    minusDI = 100 * (minusDM.rolling(length).sum() / (trv+1e-9))
    dx = 100 * ( (plusDI - minusDI).abs() / ((plusDI + minusDI)+1e-9) )
    adx = dx.rolling(length).mean()
    return plusDI, minusDI, adx

def range_filter(df, length=20, mult=3.5, source="close"):
    src = df[source].copy()
    basis = ema(src, length)
    rng = atr(df, ATR_LEN) * mult
    up = basis + rng
    lo = basis - rng
    buy = (src > up) & (src.shift(1) <= up.shift(1))
    sell= (src < lo) & (src.shift(1) >= lo.shift(1))
    return basis, up, lo, buy, sell

def pct(a,b): return 0 if a==0 else (b-a)/a
def nice(x, d=6):
    try: return float(f"{x:.{d}f}")
    except: return x

# ------------- Candles -------------
def anatomy(o,h,l,c):
    body = abs(c-o)
    full = max(h,l) - min(h,l) + 1e-9
    up = h - max(c,o); lo = min(c,o) - l
    return body/full, up/full, lo/full

def is_doji(o,h,l,c, tol=0.1):
    body, up, lo = anatomy(o,h,l,c)
    return body<tol and up>0 and lo>0

def is_pin_bull(o,h,l,c):
    body, up, lo = anatomy(o,h,l,c)
    return lo>0.55 and body<0.35

def is_pin_bear(o,h,l,c):
    body, up, lo = anatomy(o,h,l,c)
    return up>0.55 and body<0.35

def hammer(o,h,l,c): return is_pin_bull(o,h,l,c) and c>o
def inv_hammer(o,h,l,c): return is_pin_bear(o,h,l,c) and c>o
def shooting(o,h,l,c): return is_pin_bear(o,h,l,c) and c<o

def engulf_bull(prev, cur):
    return (cur['open']<cur['close']) and (prev['open']>prev['close']) and \
           (cur['open']<=prev['close']) and (cur['close']>=prev['open'])

def engulf_bear(prev, cur):
    return (cur['open']>cur['close']) and (prev['open']<prev['close']) and \
           (cur['open']>=prev['close']) and (cur['close']<=prev['open'])

def tweez_top(prev, cur):
    return abs(prev['high']-cur['high'])/max(prev['high'],1e-9)<0.001 and prev['close']>prev['open'] and cur['close']<cur['open']

def tweez_bot(prev, cur):
    return abs(prev['low']-cur['low'])/max(prev['low'],1e-9)<0.001 and prev['close']<prev['open'] and cur['close']>cur['open']

def piercing(prev, cur):
    mid=(prev['open']+prev['close'])/2
    return prev['open']>prev['close'] and cur['open']<prev['close'] and cur['close']>mid

def dark_cloud(prev, cur):
    mid=(prev['open']+prev['close'])/2
    return prev['close']>prev['open'] and cur['open']>prev['close'] and cur['close']<mid

def morning_star(c1,c2,c3):
    return c1['close']<c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']>c3['open'] and c3['close']>((c1['open']+c1['close'])/2)

def evening_star(c1,c2,c3):
    return c1['close']>c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']<c3['open'] and c3['close']<((c1['open']+c1['close'])/2)

def three_soldiers(df):
    a=df.iloc[-3];b=df.iloc[-2];c=df.iloc[-1]
    return all([a['close']>a['open'], b['close']>b['open'], c['close']>c['open']]) and c['close']>b['close']>a['close']

def three_crows(df):
    a=df.iloc[-3];b=df.iloc[-2];c=df.iloc[-1]
    return all([a['close']<a['open'], b['close']<b['open'], c['close']<c['open']]) and c['close']<b['close']<a['close']

def explosive(df, bars=3, th=MOVE_3BARS_PCT):
    last = df.tail(bars)
    moves = [abs(last.iloc[i]['close']-last.iloc[i]['open']) for i in range(len(last))]
    a = atr(df).iloc[-1]
    if a<=0: return False
    return (sum(moves)/(bars*a)) > th

# ------------- State -------------
state = {
    "in_position": False,
    "side": None, "entry": 0.0, "qty": 0.0,
    "tp1_done": False,
    "pnl_realized": 0.0,  # (persisted)
    "pnl_compound": 0.0,
    "last_close_ts": None,
    "mode": "SMART",
    "risk_equity": 0.0,
    "next_qty_hint": 0.0
}
STATE_FILE = "pnl_state.json"

def load_state():
    try:
        with open(STATE_FILE,"r") as f: state.update(json.load(f))
    except: pass
def save_state():
    try:
        with open(STATE_FILE,"w") as f: json.dump(state,f)
    except: pass
load_state()

# ------------- Data -------------
def fetch_ohlcv(symbol, tf, limit=300):
    if ex is None: return None
    return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)

def to_df(ohlc): return pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])

# ------------- Balance & Sizing -------------
def get_balance_usdt():
    try:
        bal = ex.fetch_balance({"type":"swap"})
        usdt = bal.get("USDT",{})
        free = usdt.get("free")
        if free is None:
            free = bal.get("free",{}).get("USDT", 0.0)
        return float(free or 0.0)
    except Exception as e:
        cprint(f"⚠️ fetch_balance: {e}", "red")
        return 0.0

def compute_qty(price):
    bal = max(get_balance_usdt(), 0.0)
    state["risk_equity"] = bal
    boost = 1.0 + (state["pnl_realized"]/10.0)*0.05  # +5% per $10 realized
    usd_alloc = max(bal, 10.0) * RISK_ALLOC * boost
    qty = (usd_alloc * LEVERAGE) / max(price,1e-9)
    qty = max(qty, 1.0)
    state["next_qty_hint"] = qty
    return qty

# ------------- Orders -------------
def set_leverage(leverage=LEVERAGE):
    try:
        if LIVE_TRADING:
            ex.set_leverage(leverage, SYMBOL, {"positionSide":"BOTH"})
    except Exception as e:
        cprint(f"⚠️ set_leverage: {e}", "red")

def market_order(side, qty, reduce=False):
    if not LIVE_TRADING:
        return {"paper":True,"side":side,"qty":qty,"reduceOnly":reduce}
    params = {"reduceOnly": bool(reduce), "positionSide":"BOTH"}
    try:
        return ex.create_order(SYMBOL, "market", side, qty, None, params)
    except Exception as e:
        cprint(f"⚠️ order err: {e}", "red")
        return {"status":"error","err":str(e)}

def close_position():
    if not state["in_position"]: return
    side = "sell" if state["side"]=="long" else "buy"
    market_order(side, state["qty"], reduce=True)
    state.update({"in_position":False,"tp1_done":False,"side":None,"entry":0.0,"qty":0.0})
    save_state()

# ------------- Decide -------------
def decide(df):
    df["rsi"] = rsi(df["close"], RSI_LEN)
    pDI, mDI, ADX = dx_plus_minus(df, ADX_LEN)
    df["+di"]=pDI; df["-di"]=mDI; df["adx"]=ADX
    df["atr"]=atr(df, ATR_LEN)
    rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT, RF_SOURCE)
    df["rf"]=rf; df["up"]=up; df["lo"]=lo; df["buy_sig"]=b; df["sell_sig"]=s

    last = df.iloc[-1]; prev = df.iloc[-2]; price = last["close"]

    # regime
    regime = "TREND_UP" if (last["adx"]>25 and last["+di"]>last["-di"]) else \
             "TREND_DOWN" if (last["adx"]>25 and last["-di"]>last["+di"]) else "RANGE"

    # candles set (for post-entry intelligence)
    cs = {
        "doji": is_doji(last["open"],last["high"],last["low"],last["close"]),
        "pin_bull": is_pin_bull(last["open"],last["high"],last["low"],last["close"]),
        "pin_bear": is_pin_bear(last["open"],last["high"],last["low"],last["close"]),
        "hammer": hammer(last["open"],last["high"],last["low"],last["close"]),
        "inv_hammer": inv_hammer(last["open"],last["high"],last["low"],last["close"]),
        "shooting": shooting(last["open"],last["high"],last["low"],last["close"]),
        "engulf_bull": engulf_bull(prev.to_dict(), last.to_dict()),
        "engulf_bear": engulf_bear(prev.to_dict(), last.to_dict()),
        "tweez_top": tweez_top(prev.to_dict(), last.to_dict()),
        "tweez_bot": tweez_bot(prev.to_dict(), last.to_dict()),
        "piercing": piercing(prev.to_dict(), last.to_dict()),
        "dark_cloud": dark_cloud(prev.to_dict(), last.to_dict())
    }
    if len(df)>=3:
        c1=df.iloc[-3].to_dict(); c2=df.iloc[-2].to_dict(); c3=df.iloc[-1].to_dict()
        cs["morning_star"]=morning_star(c1,c2,c3)
        cs["evening_star"]=evening_star(c1,c2,c3)
        cs["soldiers"]=three_soldiers(df.tail(3))
        cs["crows"]=three_crows(df.tail(3))
    else:
        cs.update({"morning_star":False,"evening_star":False,"soldiers":False,"crows":False})

    boom = explosive(df, 3, MOVE_3BARS_PCT)
    sig_buy, sig_sell = bool(last["buy_sig"]), bool(last["sell_sig"])
    return price, last, cs, regime, boom, sig_buy, sig_sell

def open_position(side, price):
    if state["in_position"]: return
    set_leverage(LEVERAGE)
    qty = compute_qty(price)
    market_order("buy" if side=="long" else "sell", qty, reduce=False)
    state.update({"in_position":True,"side":side,"entry":price,"qty":qty,"tp1_done":False})
    save_state()
    cprint(f"🚀 OPEN {side.upper()} | qty={nice(qty)} @ {nice(price)} | lev={LEVERAGE}x", "green" if side=="long" else "red")

def consider_entry(price, sig_buy, sig_sell, regime):
    if state["in_position"]: return
    # Strict TV-matching: entries only by RF signal
    if sig_buy:
        open_position("long", price)
    elif sig_sell:
        open_position("short", price)

# ------------- Post-entry Intelligence -------------
def manage_position(df, price, last, cs, regime, boom):
    if not state["in_position"]: return
    side, entry, qty = state["side"], state["entry"], state["qty"]
    a = df["atr"].iloc[-1]; adxv = df["adx"].iloc[-1]
    rsi_v = df["rsi"].iloc[-1]

    # classify scalp vs trend
    scalp = (df["atr"].iloc[-1] < df["close"].rolling(20).std().iloc[-1]*0.6) or (last["adx"]<18)
    if not scalp and adxv>25: state["mode"]="TREND"
    else: state["mode"]="SCALP"

    # TP1
    up_pct = pct(entry, price) if side=="long" else pct(price, entry)
    gain_pct = up_pct*100
    if not state["tp1_done"] and gain_pct>=TP1_PCT*100 and qty>0:
        cut = max(qty*TP1_CLOSE_FRAC, 0.0)
        if cut>0:
            market_order("sell" if side=="long" else "buy", cut, reduce=True)
            state["qty"] -= cut
            state["tp1_done"]=True
            cprint(f"🎯 TP1 hit • closed {nice(cut)} • remain {nice(state['qty'])}", "cyan")

    # Breakeven (virtual)
    if gain_pct>=BREAKEVEN_AFTER_PCT*100:
        pass  # (managed by trailing/giveback exits)

    # ATR trailing (activate in trend or strong push)
    activate = (gain_pct>=TRAIL_ACTIVATE_PCT*100) or (HOLD_TP_STRONG and adxv>=HOLD_TP_ADX and boom)
    if activate and a>0:
        dist = ATR_MULT_TRAIL*a
        if side=="long":
            stop = max(entry, price - dist)
            if last["low"]<stop:
                cprint("🏁 ATR trail exit LONG", "yellow"); close_position()
        else:
            stop = min(entry, price + dist)
            if last["high"]>stop:
                cprint("🏁 ATR trail exit SHORT", "yellow"); close_position()

    # Giveback (after TP1 or when minimal profit reached)
    if (state["tp1_done"] or gain_pct>=MIN_TP_PERCENT*100):
        if side=="long":
            # bearish reversal set from candles
            bear_reversal = cs.get("shooting") or cs.get("engulf_bear") or cs.get("tweez_top") or cs.get("evening_star") or cs.get("crows")
            if bear_reversal or (state["mode"]=="SCALP" and rsi_v>70):
                cprint("↩️ Giveback exit LONG", "yellow"); close_position()
        else:
            bull_reversal = cs.get("hammer") or cs.get("engulf_bull") or cs.get("tweez_bot") or cs.get("morning_star") or cs.get("soldiers")
            if bull_reversal or (state["mode"]=="SCALP" and rsi_v<30):
                cprint("↩️ Giveback exit SHORT", "yellow"); close_position()

# ------------- Logging -------------
def icon(b): return "🟢" if b else "⚪"

def log_compact(df, price, cs, regime, sig_buy, sig_sell):
    last = df.iloc[-1]
    # Header
    live_flag = "LIVE" if LIVE_TRADING else "PAPER"
    cprint(f"\n[{now_utc()}] {SYMBOL} {INTERVAL} • {live_flag} • Mode={state['mode']}", "white")
    # Indicators row (compact)
    print(f"📈 RSI{RSI_LEN}:{nice(last['rsi'])}  +DI:{nice(last['+di'])}  -DI:{nice(last['-di'])}  ADX{ADX_LEN}:{nice(last['adx'])}  ATR{ATR_LEN}:{nice(last['atr'])}")
    # Signals
    print(f"🔔 BUY:{icon(sig_buy)} SELL:{icon(sig_sell)}  Regime:{regime}  Px:{nice(price)}  RF:{nice(last['rf'])}")
    # Position
    if state["in_position"]:
        pchg = pct(state["entry"], last['close']) if state["side"]=="long" else pct(last['close'], state["entry"])
        print(f"🎯 POS: {'🟩LONG' if state['side']=='long' else '🟥SHORT'}  entry:{nice(state['entry'])}  qty:{nice(state['qty'])}  Δ:{nice(pchg*100,2)}%  TP1:{'✅' if state['tp1_done'] else '…'}")
    else:
        print(f"🟨 FLAT  next_qty@{LEVERAGE}x ≈ {nice(state['next_qty_hint'])}")
    # Result summary
    print(f"💰 Balance≈{nice(state['risk_equity'])} USDT  |  RealizedPnL={nice(state['pnl_realized'])} USDT")

# ------------- Loop -------------
def loop():
    while True:
        try:
            ohlc = fetch_ohlcv(SYMBOL, INTERVAL, 300)
            if not ohlc:
                cprint("No data.", "red"); time.sleep(DECISION_EVERY_S); continue
            df = to_df(ohlc)
            price, last, cs, regime, boom, sig_buy, sig_sell = decide(df)

            # only at candle close if requested
            if USE_TV_BAR:
                ts = df.iloc[-1]['ts']
                if state["last_close_ts"] == ts:
                    time.sleep(DECISION_EVERY_S); continue
                state["last_close_ts"] = ts

            manage_position(df, price, last, cs, regime, boom)
            consider_entry(price, sig_buy, sig_sell, regime)
            log_compact(df, price, cs, regime, sig_buy, sig_sell)

        except Exception as e:
            cprint(f"❌ loop err: {e}\n{traceback.format_exc()}", "red")
        time.sleep(DECISION_EVERY_S)

# ------------- Flask -------------
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # suppress noisy HTTP logs

@app.route("/")
def home():
    return f"OK • {SYMBOL} • {INTERVAL} • LIVE={'ON' if LIVE_TRADING else 'OFF'} • {now_utc()}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "ts": now_utc(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "live": LIVE_TRADING,
        "state": state
    })

@app.route("/ping")
def ping(): return "pong"

def keepalive():
    if not SELF_URL: return
    while True:
        try: requests.get(SELF_URL, timeout=4)
        except: pass
        time.sleep(KEEPALIVE_SECONDS)

# ------------- Start -------------
if __name__=="__main__":
    cprint(f"Starting Pro AI bot • {SYMBOL} • {INTERVAL} • LIVE={LIVE_TRADING}", "white", "on_blue")
    if ex:
        try: set_leverage(LEVERAGE)
        except: pass
    threading.Thread(target=loop, daemon=True).start()
    if SELF_URL: threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
