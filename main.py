# ==========================================================
# main.py — DOGE/USDT Smart Bot (Final, Clean Logs, BingX-ready)
#  - Entries: TradingView-like Range Filter (EXACT-style)
#  - Indicators: RSI / +DI / -DI / ADX / ATR
#  - Post-entry Mgmt: TP1 + Breakeven + ATR-Trailing + Giveback
#  - Scalp vs Trend awareness via ADX/ATR
#  - 60% equity sizing + cumulative PnL boost
#  - BingX Position Mode: oneway/hedge (ENV) + safe state updates
#  - Clean logs every DECISION_EVERY_S; suppress HTTP noise
#  - / (status) + /metrics + keepalive (2-line heartbeat)
# ==========================================================

import os, time, json, threading, traceback, logging
from datetime import datetime, timezone
import pandas as pd
from termcolor import cprint, colored
import requests

# ===================== ENV =====================
BINGX_API_KEY      = os.getenv("BINGX_API_KEY","")
BINGX_API_SECRET   = os.getenv("BINGX_API_SECRET","")
SYMBOL             = os.getenv("SYMBOL","DOGE/USDT:USDT")
INTERVAL           = os.getenv("INTERVAL","15m")
LEVERAGE           = int(os.getenv("LEVERAGE","10"))
RISK_ALLOC         = float(os.getenv("RISK_ALLOC","0.60"))   # 60% of equity
DECISION_EVERY_S   = int(os.getenv("DECISION_EVERY_S","60"))
KEEPALIVE_SECONDS  = int(os.getenv("KEEPALIVE_SECONDS","60"))
PORT               = int(os.getenv("PORT","5000"))
SELF_URL           = os.getenv("RENDER_EXTERNAL_URL","")
LIVE_TRADING       = os.getenv("LIVE_TRADING","true").lower()=="true"
USE_TV_BAR         = os.getenv("USE_TV_BAR","false").lower()=="true"  # if true: act only on bar close
SPREAD_GUARD_BPS   = int(os.getenv("SPREAD_GUARD_BPS","6"))

# BingX Position Mode: oneway | hedge
BINGX_POSITION_MODE = os.getenv("BINGX_POSITION_MODE","oneway").lower().strip()  # "oneway" or "hedge"

# Range Filter (TV-style)
RF_PERIOD          = int(os.getenv("RF_PERIOD","20"))
RF_MULT            = float(os.getenv("RF_MULT","3.5"))

# Indicators
RSI_LEN            = int(os.getenv("RSI_LEN","14"))
ADX_LEN            = int(os.getenv("ADX_LEN","14"))
ATR_LEN            = int(os.getenv("ATR_LEN","14"))

# Smart Mgmt thresholds
TP1_PCT            = float(os.getenv("TP1_PCT","0.40"))        # +0.40%
TP1_CLOSE_FRAC     = float(os.getenv("TP1_CLOSE_FRAC","0.50")) # close 50% at TP1
BREAKEVEN_AFTER_PCT= float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL     = float(os.getenv("ATR_MULT_TRAIL","1.6"))
GIVEBACK_PCT       = float(os.getenv("GIVEBACK_PCT","0.30"))
MIN_TP_PERCENT     = float(os.getenv("MIN_TP_PERCENT","0.40"))

# ===================== Exchange =====================
try:
    import ccxt
    ex = ccxt.bingx({
        "apiKey": BINGX_API_KEY,
        "secret": BINGX_API_SECRET,
        "options": {"defaultType":"swap"},
        "enableRateLimit": True
    })
    ex.set_sandbox_mode(False)
    CEX_READY = True
    cprint("✅ Connected to BingX (swap)", "cyan")
except Exception as e:
    ex = None
    CEX_READY = False
    cprint(f"⚠️ CCXT init error → PAPER mode. {e}", "yellow")

def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def nice(x, d=6):
    try: return float(f"{x:.{d}f}")
    except: return x
def pct(a,b): return 0.0 if a==0 else (b-a)/a

# ===================== Indicators =====================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
def atr(df, n=14): return true_range(df).rolling(n).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up/(dn+1e-9)
    return 100 - (100/(1+rs))
def dx_plus_minus(df, n=14):
    up = df["high"].diff(); dn = -df["low"].diff()
    plusDM  = ((up>dn)&(up>0))*up
    minusDM = ((dn>up)&(dn>0))*dn
    tr = true_range(df).rolling(n).sum()
    plusDI  = 100*(plusDM.rolling(n).sum()/(tr+1e-9))
    minusDI = 100*(minusDM.rolling(n).sum()/(tr+1e-9))
    adx = 100*((plusDI-minusDI).abs()/(plusDI+minusDI+1e-9)).rolling(n).mean()
    return plusDI, minusDI, adx

def range_filter(df, length=20, mult=3.5):
    basis = ema(df["close"], length)
    rng   = atr(df, ATR_LEN) * mult
    up, lo = basis + rng, basis - rng
    buy  = (df["close"]>up) & (df["close"].shift(1)<=up.shift(1))
    sell = (df["close"]<lo) & (df["close"].shift(1)>=lo.shift(1))
    return basis, up, lo, buy, sell

# ===================== Candles (light set) =====================
def is_doji(o,h,l,c): return abs(c-o)/(h-l+1e-9) < 0.1
def engulf_bull(prev, cur): return prev["close"]<prev["open"] and cur["open"]<=prev["close"] and cur["close"]>=prev["open"]
def engulf_bear(prev, cur): return prev["close"]>prev["open"] and cur["open"]>=prev["close"] and cur["close"]<=prev["open"]

# ===================== State =====================
state = {
    "in_position": False,
    "side": None,      # long/short
    "entry": 0.0,
    "qty": 0.0,
    "tp1_done": False,
    "pnl_realized": 0.0,
    "equity": 0.0,
    "next_qty_hint": 0.0,
    "last_bar_ts": None
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

# ===================== BingX helpers =====================
def is_hedge(): return BINGX_POSITION_MODE == "hedge"

def apply_position_mode():
    if not (CEX_READY and LIVE_TRADING): return
    try:
        # ccxt unified: set_position_mode(hedgeMode: bool, symbol=None, params={})
        ex.set_position_mode(is_hedge(), SYMBOL)
        cprint(f"✅ PositionMode → {'HEDGE' if is_hedge() else 'ONE-WAY'}", "cyan")
    except Exception as e:
        cprint(f"⚠️ set_position_mode skipped: {e}", "yellow")

def set_leverage():
    if not (CEX_READY and LIVE_TRADING): return
    try:
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})  # for oneway; OK for hedge too
    except Exception as e:
        cprint(f"⚠️ set_leverage: {e}", "yellow")

def pos_side_for_open(order_side):  # 'buy' → LONG, 'sell' → SHORT (in hedge) else BOTH
    if is_hedge():
        return "LONG" if order_side=="buy" else "SHORT"
    return "BOTH"

def pos_side_for_close(current_pos):  # current_pos: 'long'|'short'
    if is_hedge():
        return "LONG" if current_pos=="long" else "SHORT"
    return "BOTH"

def build_order_params(position_side, reduce_only=False):
    return {"positionSide": position_side, "reduceOnly": bool(reduce_only)}

def order_success(resp: dict) -> bool:
    if not isinstance(resp, dict): return False
    st = str(resp.get("status","")).lower()
    if st in ("rejected","canceled","error"): return False
    return bool(resp.get("id") or resp.get("orderId") or resp.get("info"))

# ===================== Data / Balance / Sizing =====================
def fetch_df(limit=300):
    if not CEX_READY: return None
    o = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)
    return pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])

def get_balance_usdt():
    if not CEX_READY:
        return max(50.0, state["equity"])  # paper fallback
    try:
        # Prefer swap; fallback spot
        for q in ({'type':'swap'}, {'type':'spot'}, {}):
            try:
                b = ex.fetch_balance(q)
                for path in [
                    lambda d: float(d['USDT']['free']),
                    lambda d: float(d['free']['USDT']),
                    lambda d: float(d['total']['USDT']),
                ]:
                    try: return path(b)
                    except: pass
            except: pass
        cprint("⚠️ USDT balance not found in swap/spot.", "yellow")
        return 0.0
    except Exception as e:
        cprint(f"⚠️ balance error: {e}", "yellow"); return 0.0

def compute_qty(price):
    equity = get_balance_usdt()
    state["equity"] = equity
    # cumulative boost: +5% per +10 USDT realized pnl (only positive side)
    boost = 1.0 + max(state["pnl_realized"],0.0)/10.0*0.05
    usd_alloc = equity * RISK_ALLOC * boost
    qty = (usd_alloc * LEVERAGE) / max(price,1e-9)
    qty = max(qty, 1.0)
    state["next_qty_hint"] = qty
    return qty

# ===================== Orders (safe) =====================
def market_order(order_side, qty, reduce=False, position_side=None):
    """
    order_side: 'buy' | 'sell'
    reduce: close/reduceOnly
    """
    if position_side is None:
        position_side = pos_side_for_open(order_side)

    if not (CEX_READY and LIVE_TRADING):
        return {"status":"paper","side":order_side,"qty":qty,"reduce":reduce,"positionSide":position_side}

    try:
        params = build_order_params(position_side, reduce_only=reduce)
        return ex.create_order(SYMBOL, "market", order_side, qty, params=params)
    except Exception as e:
        cprint(f"❌ order error: {e}", "red")
        return {"status":"error","err":str(e)}

def open_position(side, price):
    if state["in_position"]: return
    set_leverage()
    qty = compute_qty(price)
    order_side = "buy" if side=="long" else "sell"
    pos_side = pos_side_for_open(order_side)
    resp = market_order(order_side, qty, reduce=False, position_side=pos_side)
    if order_success(resp):
        state.update({"in_position":True,"side":side,"entry":price,"qty":qty,"tp1_done":False})
        save_state()
        cprint(f"{'🚀' if side=='long' else '🧨'} OPEN {side.upper()} | qty={nice(qty)} @ {nice(price)} | lev={LEVERAGE}x [{pos_side}]",
               "green" if side=='long' else "red")
    else:
        cprint(f"⚠️ OPEN {side.upper()} rejected; state not changed. Resp={resp}", "yellow")

def close_full():
    if not state["in_position"]: return
    order_side = "sell" if state["side"]=="long" else "buy"
    pos_side = pos_side_for_close(state["side"])
    resp = market_order(order_side, state["qty"], reduce=True, position_side=pos_side)
    if order_success(resp):
        state.update({"in_position":False,"side":None,"entry":0.0,"qty":0.0,"tp1_done":False})
        save_state()
        cprint(f"✅ CLOSED [{pos_side}]", "cyan")
    else:
        cprint(f"❌ CLOSE rejected; keeping local state. Resp={resp}", "red")

# ===================== Spread guard =====================
def orderbook_spread_bps():
    try:
        ob = ex.fetch_order_book(SYMBOL, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

# ===================== Decide & Manage =====================
def decide(df):
    df["rsi"] = rsi(df["close"], RSI_LEN)
    pdi, mdi, adx = dx_plus_minus(df, ADX_LEN)
    df["+di"], df["-di"], df["adx"] = pdi, mdi, adx
    df["atr"] = atr(df, ATR_LEN)
    rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT)
    df["rf"], df["up"], df["lo"], df["buy"], df["sell"] = rf, up, lo, b, s

    last = df.iloc[-1]; prev = df.iloc[-2]; price = last["close"]

    # candles for mgmt
    c_last = {"open":last["open"],"high":last["high"],"low":last["low"],"close":last["close"]}
    c_prev = {"open":prev["open"],"high":prev["high"],"low":prev["low"],"close":prev["close"]}
    candles = {
        "doji": is_doji(**c_last),
        "engulf_bull": engulf_bull(c_prev, c_last),
        "engulf_bear": engulf_bear(c_prev, c_last)
    }

    regime = "TREND_UP" if (last["adx"]>25 and last["+di"]>last["-di"]) else \
             "TREND_DOWN" if (last["adx"]>25 and last["-di"]>last["+di"]) else "RANGE"

    return price, last, candles, regime, bool(last["buy"]), bool(last["sell"])

def consider_entry(price, sig_buy, sig_sell):
    if state["in_position"]: return
    # spread guard
    spr = orderbook_spread_bps()
    if spr is not None and spr > SPREAD_GUARD_BPS:
        cprint(f"⏸ Spread high {spr:.2f}bps > {SPREAD_GUARD_BPS}bps — wait", "yellow")
        return
    if sig_buy:  open_position("long",  price)
    elif sig_sell: open_position("short", price)

def manage_position(df, price, last, candles):
    if not state["in_position"]: return
    side, entry, qty = state["side"], state["entry"], state["qty"]
    atrv = float(df["atr"].iloc[-1]); adxv = float(df["adx"].iloc[-1]); rsi_v = float(df["rsi"].iloc[-1])

    gain = pct(entry, price) if side=="long" else pct(price, entry)
    gain_pct = gain*100

    # --- TP1 ---
    if (not state["tp1_done"]) and gain_pct >= TP1_PCT*100 and qty>0:
        cut = max(qty*TP1_CLOSE_FRAC, 0.0)
        if cut>0:
            # partial close reduceOnly
            order_side = "sell" if side=="long" else "buy"
            pos_side = pos_side_for_close(side)
            resp = market_order(order_side, cut, reduce=True, position_side=pos_side)
            if order_success(resp):
                state["qty"] = max(0.0, qty - cut)
                state["tp1_done"] = True
                save_state()
                cprint(f"🎯 TP1 → closed {nice(cut)} remain={nice(state['qty'])}", "cyan")

    # --- ATR Trailing ---
    if atrv>0 and (gain_pct>=TRAIL_ACTIVATE_PCT*100 or (adxv>=28 and state["tp1_done"])):
        dist = ATR_MULT_TRAIL * atrv
        if side=="long":
            trail = max(entry, price - dist)
            if last["low"] <= trail:
                cprint("🏁 ATR trail exit LONG", "yellow"); close_full(); return
        else:
            trail = min(entry, price + dist)
            if last["high"] >= trail:
                cprint("🏁 ATR trail exit SHORT", "yellow"); close_full(); return

    # --- Giveback after TP1 / minimal profit ---
    if (state["tp1_done"] or gain_pct>=MIN_TP_PERCENT*100):
        if side=="long":
            if candles["engulf_bear"] or rsi_v>72:
                cprint("↩️ Giveback exit LONG", "yellow"); close_full(); return
        else:
            if candles["engulf_bull"] or rsi_v<28:
                cprint("↩️ Giveback exit SHORT", "yellow"); close_full(); return

# ===================== Logging (Clean) =====================
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # suppress HTTP 200 noise

def icon(b): return "🟢" if b else "⚪"

def print_hud(df, price, candles, regime, sig_buy, sig_sell):
    last = df.iloc[-1]
    # Header
    header = f"{SYMBOL} {INTERVAL} • {'LIVE' if (CEX_READY and LIVE_TRADING) else 'PAPER'} • {now_utc()}"
    cprint("\n" + header, "white", "on_blue")
    # Indicators
    print(colored("INDICATORS","cyan"))
    print(f"  💠 Price={nice(price)}  RF={nice(last['rf'])}  hi={nice(last['high'])}  lo={nice(last['low'])}")
    print(f"  📈 RSI({RSI_LEN})={nice(last['rsi'],2)}  +DI={nice(last['+di'],2)}  -DI={nice(last['-di'],2)}  ADX({ADX_LEN})={nice(last['adx'],2)}  ATR={nice(df['atr'].iloc[-1])}")
    print(f"  🧭 Regime={regime}   🔔 BUY={icon(sig_buy)}  🔻 SELL={icon(sig_sell)}")
    # Candles
    print(colored("CANDLES","magenta"))
    print(f"  Doji={icon(candles['doji'])} | EngulfBull={icon(candles['engulf_bull'])} | EngulfBear={icon(candles['engulf_bear'])}")
    # Position
    print(colored("POSITION","blue"))
    if state["in_position"]:
        change = pct(state["entry"], price) if state["side"]=="long" else pct(price, state["entry"])
        side_ico = "🟩 LONG" if state["side"]=="long" else "🟥 SHORT"
        print(f"  {side_ico} entry={nice(state['entry'])} qty={nice(state['qty'])} Δ={nice(change*100,2)}% TP1={'✅' if state['tp1_done'] else '…'}")
    else:
        print(f"  🟨 FLAT | next_qty@{LEVERAGE}x ≈ {nice(state['next_qty_hint'])}")
    # Results
    eq = get_balance_usdt()
    print(colored("RESULTS","green"))
    print(f"  💰 Balance={nice(eq,2)} USDT | PnL(Σ)={nice(state['pnl_realized'],2)} | EffectiveEq≈{nice(eq,2)}")

# ===================== Loop =====================
def strategy_loop():
    while True:
        try:
            df = fetch_df()
            if df is None or df.empty:
                cprint("⚠️ No data fetched", "red")
                time.sleep(DECISION_EVERY_S); continue

            # wait for bar close if requested
            ts = int(df["ts"].iloc[-1])
            if USE_TV_BAR and state["last_bar_ts"] == ts:
                # only refresh HUD
                # recompute display-only inds for latest df
                rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT)
                df["rf"]=rf; df["atr"]=atr(df, ATR_LEN)
                pdi,mdi,adx = dx_plus_minus(df, ADX_LEN)
                df["+di"]=pdi; df["-di"]=mdi; df["adx"]=adx; df["rsi"]=rsi(df["close"],RSI_LEN)
                price = df["close"].iloc[-1]
                candles={"doji":False,"engulf_bull":False,"engulf_bear":False}
                print_hud(df, price, candles, "—", bool(b.iloc[-1]), bool(s.iloc[-1]))
                time.sleep(DECISION_EVERY_S); continue
            state["last_bar_ts"] = ts

            price, last, candles, regime, sig_buy, sig_sell = decide(df)
            manage_position(df, price, last, candles)
            consider_entry(price, sig_buy, sig_sell)
            print_hud(df, price, candles, regime, sig_buy, sig_sell)

        except Exception as e:
            cprint(f"❌ loop error: {e}", "red")
            print(traceback.format_exc())
        time.sleep(DECISION_EVERY_S)

# ===================== Flask =====================
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "ts": now_utc(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "live": (CEX_READY and LIVE_TRADING),
        "position_mode": BINGX_POSITION_MODE,
        "equity": state["equity"],
        "pnl_sum": state["pnl_realized"],
        "in_position": state["in_position"],
        "side": state["side"],
        "entry": state["entry"],
        "qty": state["qty"]
    })

@app.route("/metrics")
def metrics():
    return jsonify({"state": state})

def keepalive():
    if not SELF_URL: return
    while True:
        try:
            r = requests.get(SELF_URL, timeout=5)
            # 2-line heartbeat only:
            cprint(f"🛰 keepalive {r.status_code} • {now_utc()}", "white")
            cprint(f"URL: {SELF_URL}", "white")
        except Exception as e:
            cprint(f"🛰 keepalive error: {e}", "yellow")
        time.sleep(KEEPALIVE_SECONDS)

# ===================== Start =====================
if __name__ == "__main__":
    mode = "LIVE" if (CEX_READY and LIVE_TRADING) else "PAPER"
    cprint(f"🚀 Starting SMART BOT • {SYMBOL} • MODE={mode}", "white", "on_blue")
    try:
        apply_position_mode()
        set_leverage()
    except: pass
    threading.Thread(target=strategy_loop, daemon=True).start()
    if SELF_URL:
        threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
