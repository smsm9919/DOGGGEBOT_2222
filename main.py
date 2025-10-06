# ==========================================================
# main.py — DOGE/USDT Smart AI Bot (FINAL)
#  - TV-matching entries (Range Filter)
#  - RSI / +DI / -DI / ADX / ATR
#  - Full trade management: TP1 + Breakeven + ATR Trailing + Giveback
#  - Scalp/Trend awareness via ADX and move strength
#  - Cumulative PnL sizing (uses 60% equity by default)
#  - Clean color logs + icons, 24/7 loop + keepalive
# ==========================================================

import os, time, json, math, threading, traceback
from datetime import datetime, timezone
import pandas as pd
from termcolor import cprint, colored
import requests

# ================= ENV =================
BINGX_API_KEY     = os.getenv("BINGX_API_KEY","")
BINGX_API_SECRET  = os.getenv("BINGX_API_SECRET","")
SYMBOL            = os.getenv("SYMBOL","DOGE/USDT:USDT")
INTERVAL          = os.getenv("INTERVAL","15m")
LEVERAGE          = int(os.getenv("LEVERAGE","10"))
RISK_ALLOC        = float(os.getenv("RISK_ALLOC","0.60"))  # 60% من الرصيد
DECISION_EVERY_S  = int(os.getenv("DECISION_EVERY_S","60"))
KEEPALIVE_SECONDS = int(os.getenv("KEEPALIVE_SECONDS","60"))
PORT              = int(os.getenv("PORT","5000"))
SELF_URL          = os.getenv("RENDER_EXTERNAL_URL","")
LIVE_TRADING      = os.getenv("LIVE_TRADING","true").lower() == "true"
USE_TV_BAR        = os.getenv("USE_TV_BAR","false").lower() == "true"  # لو true انتظر إغلاق الشمعة

# Range Filter (مطابق TV)
RF_PERIOD = int(os.getenv("RF_PERIOD","20"))
RF_MULT   = float(os.getenv("RF_MULT","3.5"))

# مؤشرات
RSI_LEN = int(os.getenv("RSI_LEN","14"))
ADX_LEN = int(os.getenv("ADX_LEN","14"))
ATR_LEN = int(os.getenv("ATR_LEN","14"))

# إدارة ذكية
TP1_PCT            = float(os.getenv("TP1_PCT","0.40"))        # هدف أول % ربح
TP1_CLOSE_FRAC     = float(os.getenv("TP1_CLOSE_FRAC","0.50")) # إغلاق جزء عند TP1
BREAKEVEN_AFTER_PCT= float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL     = float(os.getenv("ATR_MULT_TRAIL","1.6"))
GIVEBACK_PCT       = float(os.getenv("GIVEBACK_PCT","0.30"))
MIN_TP_PERCENT     = float(os.getenv("MIN_TP_PERCENT","0.40")) # لا تفعل giveback قبل هذا

# ================= Exchange =================
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
    cprint(f"⚠️ CCXT init error → Paper Mode: {e}", "yellow")

def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def nice(x, d=6):
    try: return float(f"{x:.{d}f}")
    except: return x
def pct(a,b): return 0 if a==0 else (b-a)/a

# ================= Indicators =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
def atr(df, n): return true_range(df).rolling(n).mean()
def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up/(dn+1e-9)
    return 100 - (100/(1+rs))
def dx_plus_minus(df, n):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plusDM  = ((up>dn)&(up>0))*up
    minusDM = ((dn>up)&(dn>0))*dn
    tr = true_range(df).rolling(n).sum()
    plusDI  = 100*(plusDM.rolling(n).sum()/(tr+1e-9))
    minusDI = 100*(minusDM.rolling(n).sum()/(tr+1e-9))
    adx = 100*((plusDI-minusDI).abs()/(plusDI+minusDI+1e-9)).rolling(n).mean()
    return plusDI, minusDI, adx

def range_filter(df, length=20, mult=3.5):
    basis = ema(df["close"], length)
    rng   = atr(df, ATR_LEN)*mult
    up, lo = basis+rng, basis-rng
    buy  = (df["close"]>up) & (df["close"].shift(1)<=up.shift(1))
    sell = (df["close"]<lo) & (df["close"].shift(1)>=lo.shift(1))
    return basis, up, lo, buy, sell

# ================= Candles (مختارة) =================
def is_doji(o,h,l,c): return abs(c-o)/(h-l+1e-9) < 0.1
def engulf_bull(prev, cur): return prev["close"]<prev["open"] and cur["close"]>=prev["open"] and cur["open"]<=prev["close"]
def engulf_bear(prev, cur): return prev["close"]>prev["open"] and cur["close"]<=prev["open"] and cur["open"]>=prev["close"]

# ================= State =================
state = {
    "in_position": False,
    "side": None,          # long/short
    "entry": 0.0,
    "qty": 0.0,
    "tp1_done": False,
    "pnl_realized": 0.0,   # تراكمي
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
        with open(STATE_FILE,"w") as f: json.dump(state, f)
    except: pass
load_state()

# ================= Data / Balance =================
def fetch_df(limit=300):
    if not CEX_READY: return None
    o = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
    return df

def get_balance_usdt():
    if not CEX_READY:  # Paper fallback
        return max(50.0, state["equity"])
    try:
        bal = ex.fetch_balance()
        return float(bal["USDT"]["free"])
    except Exception as e:
        cprint(f"⚠️ balance error: {e}", "yellow")
        return 0.0

# ================= Sizing =================
def compute_qty(price):
    equity = get_balance_usdt()
    state["equity"] = equity
    # ربح تراكمي: زود 5% لكل 10 USDT ربح محقق
    boost = 1.0 + max(state["pnl_realized"],0.0)/10.0*0.05
    usd_alloc = equity * RISK_ALLOC * boost
    qty = (usd_alloc * LEVERAGE) / max(price,1e-9)
    state["next_qty_hint"] = max(qty, 0.0)
    return max(qty, 1.0)

# ================= Orders =================
def set_leverage():
    if not (CEX_READY and LIVE_TRADING): return
    try:
        # BingX: ممكن يحتاج side=BOTH لعقود اتجاه واحد
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
    except Exception as e:
        cprint(f"⚠️ set_leverage: {e}", "yellow")

def market_order(side, qty, reduce=False):
    if not (CEX_READY and LIVE_TRADING):
        return {"status":"paper","side":side,"qty":qty,"reduce":reduce}
    try:
        typ = "sell" if side=="sell" else "buy"
        params = {"reduceOnly": True} if reduce else {"reduceOnly": False}
        return ex.create_order(SYMBOL, "market", typ, qty, params=params)
    except Exception as e:
        cprint(f"❌ order error: {e}", "red")
        return {"status":"error","err":str(e)}

def close_full():
    if not state["in_position"]: return
    # عكس اتجاه الصفقة وبـ reduceOnly=True
    side = "sell" if state["side"]=="long" else "buy"
    market_order(side, state["qty"], reduce=True)
    state.update({"in_position":False,"side":None,"entry":0.0,"qty":0.0,"tp1_done":False})
    save_state()

# ================= Decision =================
def decide(df):
    df["rsi"] = rsi(df["close"], RSI_LEN)
    pdi, mdi, adx = dx_plus_minus(df, ADX_LEN)
    df["+di"], df["-di"], df["adx"] = pdi, mdi, adx
    df["atr"] = atr(df, ATR_LEN)
    rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT)
    df["rf"], df["up"], df["lo"], df["buy"], df["sell"] = rf, up, lo, b, s

    last = df.iloc[-1]; prev = df.iloc[-2]
    price = last["close"]

    # شموع أساسية
    c_last = {"open":last["open"],"high":last["high"],"low":last["low"],"close":last["close"]}
    c_prev = {"open":prev["open"],"high":prev["high"],"low":prev["low"],"close":prev["close"]}
    candles = {
        "doji": is_doji(**c_last),
        "engulf_bull": engulf_bull(c_prev, c_last),
        "engulf_bear": engulf_bear(c_prev, c_last)
    }

    # نظام السوق
    regime = "TREND_UP" if (last["adx"]>25 and last["+di"]>last["-di"]) else \
             "TREND_DOWN" if (last["adx"]>25 and last["-di"]>last["+di"]) else "RANGE"

    return price, last, candles, regime, bool(last["buy"]), bool(last["sell"])

def consider_entry(price, sig_buy, sig_sell):
    if state["in_position"]: return
    if sig_buy:
        qty = compute_qty(price)
        set_leverage()
        market_order("buy", qty, reduce=False)
        state.update({"in_position":True,"side":"long","entry":price,"qty":qty,"tp1_done":False})
        save_state()
        cprint(f"🚀 OPEN LONG | qty={nice(qty)} @ {nice(price)} | lev={LEVERAGE}x","green")
    elif sig_sell:
        qty = compute_qty(price)
        set_leverage()
        market_order("sell", qty, reduce=False)
        state.update({"in_position":True,"side":"short","entry":price,"qty":qty,"tp1_done":False})
        save_state()
        cprint(f"🧨 OPEN SHORT | qty={nice(qty)} @ {nice(price)} | lev={LEVERAGE}x","red")

def manage_position(df, price, last):
    if not state["in_position"]: return
    side = state["side"]; entry = state["entry"]; qty = state["qty"]
    atrv = float(df["atr"].iloc[-1]); adxv = float(df["adx"].iloc[-1])

    # ربح/خسارة نسبةً للاتجاه
    gain = pct(entry, price) if side=="long" else pct(price, entry)
    gain_pct = gain*100

    # --- TP1 ---
    if (not state["tp1_done"]) and gain_pct >= TP1_PCT*100:
        cut = max(qty*TP1_CLOSE_FRAC, 0.0)
        if cut>0:
            market_order("sell" if side=="long" else "buy", cut, reduce=True)
            state["qty"] = max(0.0, qty - cut)
            state["tp1_done"] = True
            cprint(f"🎯 TP1 hit → closed {nice(cut)} remain={nice(state['qty'])}", "cyan")

    # --- Breakeven (نمنع الخسارة بعد مكسب معقول) ---
    # نطبق منطقيًا بتتبع الخروج؛ (بدون تعديل أمر وقف فعليًا لتبسيط التكامل).
    # يمكن تحويله إلى أمر stop-market لاحقًا إن أردت.
    breakeven_ready = gain_pct >= BREAKEVEN_AFTER_PCT*100

    # --- ATR Trailing ---
    if atrv>0 and (gain_pct>=TRAIL_ACTIVATE_PCT*100 or (adxv>28 and state["tp1_done"])):
        dist = ATR_MULT_TRAIL*atrv
        if side=="long":
            trail_stop = max(entry, price - dist)
            if last["low"] <= trail_stop:
                cprint(f"🏁 ATR Trailing exit LONG @ {nice(price)}", "yellow")
                close_full()
                return
        else:
            trail_stop = min(entry, price + dist)
            if last["high"] >= trail_stop:
                cprint(f"🏁 ATR Trailing exit SHORT @ {nice(price)}", "yellow")
                close_full()
                return

    # --- Giveback ---
    if state["tp1_done"] and gain_pct >= MIN_TP_PERCENT*100:
        # لو رجع السعر بنسبة GIVEBACK من القمة/القاع النسبي اخرج
        if side=="long":
            target = entry*(1 + (gain - GIVEBACK_PCT))
            if price <= target:
                cprint("↩️ Giveback exit LONG", "yellow")
                close_full()
        else:
            target = entry*(1 - (gain - GIVEBACK_PCT))
            if price >= target:
                cprint("↩️ Giveback exit SHORT", "yellow")
                close_full()

# ================= Logging (HUD) =================
def icon(b): return "🟢" if b else "⚪"
def print_hud(df, price, candles, regime, sig_buy, sig_sell):
    last = df.iloc[-1]
    print()
    header = f"{SYMBOL} • {INTERVAL} • {'LIVE' if (CEX_READY and LIVE_TRADING) else 'PAPER'} • {now_utc()}"
    cprint(header, "white", "on_blue")

    # مؤشرات
    print(colored("INDICATORS","cyan"))
    print(f"  💠 Price={nice(price)}  RF={nice(last['rf'])}  hi={nice(last['high'])}  lo={nice(last['low'])}")
    print(f"  📈 RSI({RSI_LEN})={nice(last['rsi'],2)}  +DI={nice(last['+di'],2)}  -DI={nice(last['-di'],2)}  ADX({ADX_LEN})={nice(last['adx'],2)}  ATR={nice(df['atr'].iloc[-1])}")
    print(f"  🧭 Regime={regime}   🔔 BUY={icon(sig_buy)}  🔻 SELL={icon(sig_sell)}")

    # شموع
    print(colored("CANDLES","magenta"))
    print(f"  Doji={icon(candles['doji'])} | EngulfBull={icon(candles['engulf_bull'])} | EngulfBear={icon(candles['engulf_bear'])}")

    # مركز
    print(colored("POSITION","blue"))
    if state["in_position"]:
        change = pct(state["entry"], price) if state["side"]=="long" else pct(price, state["entry"])
        side_ico = "🟩 LONG" if state["side"]=="long" else "🟥 SHORT"
        print(f"  {side_ico}  entry={nice(state['entry'])}  qty={nice(state['qty'])}  Δ={nice(change*100,2)}%  TP1={icon(state['tp1_done'])}")
    else:
        print(f"  🟨 FLAT  | next_qty_hint≈{nice(state['next_qty_hint'])} @ {LEVERAGE}x")

    # نتائج
    equity = get_balance_usdt()
    print(colored("RESULTS","green"))
    print(f"  💰 Balance={nice(equity,2)} USDT  |  PnL(Σ)={nice(state['pnl_realized'],2)}  |  EffectiveEq≈{nice(equity,2)}")

# ================= Loop =================
def strategy_loop():
    while True:
        try:
            df = fetch_df()
            if df is None or df.empty:
                cprint("⚠️ No data fetched", "red")
                time.sleep(DECISION_EVERY_S); continue

            # إغلاق الشمعة (لو USE_TV_BAR=true)
            bar_ts = int(df["ts"].iloc[-1])
            if USE_TV_BAR and state["last_bar_ts"] == bar_ts:
                # HUD فقط
                price = df["close"].iloc[-1]
                # مؤشرات للعرض فقط
                rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT)
                df["rf"]=rf; df["atr"]=atr(df, ATR_LEN)
                pdi,mdi,adx = dx_plus_minus(df, ADX_LEN)
                df["+di"]=pdi; df["-di"]=mdi; df["adx"]=adx; df["rsi"]=rsi(df["close"],RSI_LEN)
                candles = {"doji": False, "engulf_bull": False, "engulf_bear": False}
                print_hud(df, price, candles, "—", bool(b.iloc[-1]), bool(s.iloc[-1]))
                time.sleep(DECISION_EVERY_S); continue
            state["last_bar_ts"] = bar_ts

            price, last, candles, regime, sig_buy, sig_sell = decide(df)

            # إدارة الصفقة الحالية
            manage_position(df, price, last)

            # دخول جديد (توافق TV)
            consider_entry(price, sig_buy, sig_sell)

            # HUD
            print_hud(df, price, candles, regime, sig_buy, sig_sell)

        except Exception as e:
            cprint(f"❌ loop error: {e}", "red")
            print(traceback.format_exc())
        time.sleep(DECISION_EVERY_S)

# ================= Flask =================
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "ts": now_utc(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "live": (CEX_READY and LIVE_TRADING),
        "equity": state["equity"],
        "pnl": state["pnl_realized"],
        "in_position": state["in_position"],
        "side": state["side"],
        "entry": state["entry"],
        "qty": state["qty"]
    })

@app.route("/ping")
def ping(): return "pong"

def keepalive():
    if not SELF_URL: return
    while True:
        try: requests.get(SELF_URL, timeout=5)
        except: pass
        time.sleep(KEEPALIVE_SECONDS)

# ================= Start =================
if __name__ == "__main__":
    mode = "LIVE" if (CEX_READY and LIVE_TRADING) else "PAPER"
    cprint(f"🚀 Starting SMART BOT • {SYMBOL} • MODE={mode}", "white", "on_blue")
    try: set_leverage()
    except: pass
    threading.Thread(target=strategy_loop, daemon=True).start()
    threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
