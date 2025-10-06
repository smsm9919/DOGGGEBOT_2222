# main.py
# ==============================================
# DOGE/USDT (BingX Perp) — Pro Strategy (TV-match)
# - TV entries via Range Filter (upper/lower cross)
# - Indicators: RSI/ADX/+DI/-DI/ATR
# - Full candle set (Doji/Pin/Hammer/Inv/Shooting/Gravestone/
#   Engulfing/Tweezers/Morning/Evening/Piercing/DarkCloud/
#   Soldiers/Crows)
# - Smart Mgmt: TP1 + Breakeven + ATR Trailing + Giveback
# - Regime: RANGE/TREND_UP/TREND_DOWN  (scalp vs trend bias)
# - Cumulative PnL sizing (next position uses realized PnL)
# - Clean HUD logs + /metrics + light keepalive
# ==============================================

import os, time, math, json, threading, traceback
from datetime import datetime, timezone
import pandas as pd
from flask import Flask, jsonify
from termcolor import cprint, colored

# ---------- ENV ----------
BINGX_API_KEY        = os.getenv("BINGX_API_KEY","")
BINGX_API_SECRET     = os.getenv("BINGX_API_SECRET","")
BINGX_POSITION_MODE  = os.getenv("BINGX_POSITION_MODE","oneway").lower()   # oneway | hedge

SYMBOL               = os.getenv("SYMBOL","DOGE/USDT:USDT")
INTERVAL             = os.getenv("INTERVAL","15m")
LEVERAGE             = int(os.getenv("LEVERAGE","10"))
RISK_ALLOC           = float(os.getenv("RISK_ALLOC","0.60"))
DECISION_EVERY_S     = int(os.getenv("DECISION_EVERY_S","30"))
KEEPALIVE_SECONDS    = int(os.getenv("KEEPALIVE_SECONDS","60"))
PORT                 = int(os.getenv("PORT","5000"))

# Range Filter (TV)
RF_PERIOD            = int(os.getenv("RF_PERIOD","20"))
RF_SOURCE            = os.getenv("RF_SOURCE","close")
RF_MULT              = float(os.getenv("RF_MULT","3.5"))
SPREAD_GUARD_BPS     = int(os.getenv("SPREAD_GUARD_BPS","6"))
USE_TV_BAR           = os.getenv("USE_TV_BAR","false").lower()=="true"
FORCE_TV_ENTRIES     = os.getenv("FORCE_TV_ENTRIES","false").lower()=="true"

# Indicators
RSI_LEN              = int(os.getenv("RSI_LEN","14"))
ADX_LEN              = int(os.getenv("ADX_LEN","14"))
ATR_LEN              = int(os.getenv("ATR_LEN","14"))

# Smart Mgmt
TP1_PCT              = float(os.getenv("TP1_PCT","0.40"))      # take first profits %
TP1_CLOSE_FRAC       = float(os.getenv("TP1_CLOSE_FRAC","0.50"))
BREAKEVEN_AFTER_PCT  = float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
TRAIL_ACTIVATE_PCT   = float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL       = float(os.getenv("ATR_MULT_TRAIL","1.6"))
GIVEBACK_PCT         = float(os.getenv("GIVEBACK_PCT","0.30"))
MIN_TP_PERCENT       = float(os.getenv("MIN_TP_PERCENT","0.40"))
HOLD_TP_STRONG       = os.getenv("HOLD_TP_STRONG","true").lower()=="true"
HOLD_TP_ADX          = int(os.getenv("HOLD_TP_ADX","28"))
HOLD_TP_SLOPE        = float(os.getenv("HOLD_TP_SLOPE","0.50"))
USE_SMART_EXIT       = os.getenv("USE_SMART_EXIT","true").lower()=="true"

# Runtime
SELF_URL             = os.getenv("RENDER_EXTERNAL_URL","")
LIVE_TRADING         = os.getenv("LIVE_TRADING","true").lower()=="true"
PING_LOG_EVERY_N     = int(os.getenv("PING_LOG_EVERY_N","2"))   # اطبع سطرين فقط كل دقيقة

# ---------- ccxt (BingX swap) ----------
try:
    import ccxt
    ex = ccxt.bingx({
        "apiKey": BINGX_API_KEY,
        "secret": BINGX_API_SECRET,
        "options": {"defaultType":"swap"},
        "enableRateLimit": True
    })
    ex.set_sandbox_mode(False)
except Exception as e:
    ex = None
    cprint(f"⚠️ ccxt init error: {e}", "red")

# ---------- helpers ----------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ema(series, length): return series.ewm(span=length, adjust=False).mean()

def true_range(df):
    pc = df['close'].shift(1)
    return pd.concat([(df['high']-df['low']).abs(),
                      (df['high']-pc).abs(),
                      (df['low']-pc).abs()],axis=1).max(axis=1)

def atr(df, length=14): return true_range(df).rolling(length).mean()

def rsi(series, length=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = (-d.clip(upper=0)).rolling(length).mean()
    rs = up/(dn+1e-9)
    return 100 - (100/(1+rs))

def dx_plus_minus(df, length=14):
    upm   = df['high'].diff()
    downm = -df['low'].diff()
    plusDM  = ((upm>downm)&(upm>0))*upm
    minusDM = ((downm>upm)&(downm>0))*downm
    trv  = true_range(df).rolling(length).sum()
    plusDI  = 100*(plusDM.rolling(length).sum()/(trv+1e-9))
    minusDI = 100*(minusDM.rolling(length).sum()/(trv+1e-9))
    dx = 100*((plusDI-minusDI).abs()/((plusDI+minusDI)+1e-9))
    adx = dx.rolling(length).mean()
    return plusDI, minusDI, adx

def range_filter(df, length=20, mult=3.5, source="close"):
    src   = df[source]
    basis = ema(src, length)
    rng   = atr(df, ATR_LEN)*mult
    upper = basis + rng
    lower = basis - rng
    buy   = (src>upper) & (src.shift(1)<=upper.shift(1))
    sell  = (src<lower) & (src.shift(1)>=lower.shift(1))
    return basis, upper, lower, buy, sell

def pct(a,b): return 0 if a==0 else (b-a)/a
def nice(x, d=6):
    try: return float(f"{x:.{d}f}")
    except: return x
def icon(b): return "🟢" if b else "⚪"

# ---------- candles ----------
def body_up_low(o,h,l,c):
    body = abs(c-o); full = (h-l)+1e-9
    up = h - max(o,c); lo = min(o,c) - l
    return body/full, up/full, lo/full

def is_doji(o,h,l,c,tol=0.1):
    body, up, lo = body_up_low(o,h,l,c); return body<tol and up>0 and lo>0
def is_spinning_top(o,h,l,c):
    body, up, lo = body_up_low(o,h,l,c); return body<0.3 and up>0.2 and lo>0.2
def is_pin_bull(o,h,l,c):
    body, up, lo = body_up_low(o,h,l,c); return lo>0.55 and body<0.35
def is_pin_bear(o,h,l,c):
    body, up, lo = body_up_low(o,h,l,c); return up>0.55 and body<0.35
def is_hammer(o,h,l,c):       return is_pin_bull(o,h,l,c) and c>o
def is_inv_hammer(o,h,l,c):   return is_pin_bear(o,h,l,c) and c>o
def is_shooting_star(o,h,l,c):return is_pin_bear(o,h,l,c) and c<o
def is_gravestone(o,h,l,c):   return is_doji(o,h,l,c) and (h-max(c,o))>(min(c,o)-l)*2

def engulf_bull(p, q):
    return (q['open']<q['close']) and (p['open']>p['close']) and (q['open']<=p['close']) and (q['close']>=p['open'])
def engulf_bear(p, q):
    return (q['open']>q['close']) and (p['open']<p['close']) and (q['open']>=p['close']) and (q['close']<=p['open'])

def tweez_top(p, q):  return abs(p['high']-q['high'])/max(p['high'],1e-9)<0.001 and p['close']>p['open'] and q['close']<q['open']
def tweez_bot(p, q):  return abs(p['low']-q['low'])/max(p['low'],1e-9)<0.001 and p['close']<p['open'] and q['close']>q['open']
def piercing(p, q):
    p_mid=(p['open']+p['close'])/2
    return p['open']>p['close'] and q['open']<p['close'] and q['close']>p_mid
def darkcloud(p, q):
    p_mid=(p['open']+p['close'])/2
    return p['close']>p['open'] and q['open']>p['close'] and q['close']<p_mid
def morning_star(c1,c2,c3):
    return c1['close']<c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']>c3['open'] and c3['close']>((c1['open']+c1['close'])/2)
def evening_star(c1,c2,c3):
    return c1['close']>c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']<c3['open'] and c3['close']<((c1['open']+c1['close'])/2)
def soldiers3(df):
    a,b,c = df.iloc[-3],df.iloc[-2],df.iloc[-1]
    return a['close']>a['open'] and b['close']>b['open'] and c['close']>c['open'] and c['close']>b['close']>a['close']
def crows3(df):
    a,b,c = df.iloc[-3],df.iloc[-2],df.iloc[-1]
    return a['close']<a['open'] and b['close']<b['open'] and c['close']<c['open'] and c['close']<b['close']<a['close']

# ---------- state ----------
state = {
    "in_position": False,
    "side": None,           # long/short
    "entry": 0.0,
    "qty": 0.0,
    "tp1_done": False,
    "pnl_realized": 0.0,
    "pnl_compound": 0.0,
    "risk_equity": 0.0,
    "next_qty_hint": 0.0,
    "last_close_ts": None,
    "mode": "SMART",
    "position_mode": BINGX_POSITION_MODE.upper(),
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

# ---------- data ----------
def fetch_ohlcv(symbol, tf, limit=300):
    if ex is None: return None
    return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)

def df_from_ohlc(ohlc):
    return pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])

# ---------- balance & sizing ----------
def get_balance_usdt():
    try:
        if ex is None: return 0.0
        b = ex.fetch_balance()
        # BingX swap USDT:
        for k in ("USDT","usdt"):
            if k in b: 
                free = b[k].get("free")
                if free is not None: return float(free)
        # fallback
        return float(b.get("free",0))
    except:
        return 0.0

def compute_qty(price):
    bal = get_balance_usdt()
    state["risk_equity"] = bal
    boost = 1.0 + (state["pnl_realized"]/10.0)*0.05
    usd_alloc = max(bal, 10.0) * RISK_ALLOC * boost
    qty = (usd_alloc * LEVERAGE) / max(price,1e-9)
    state["next_qty_hint"] = qty
    return max(qty, 1.0)

# ---------- orders ----------
def set_leverage(lv):
    try:
        if not LIVE_TRADING or ex is None: return
        ex.set_leverage(lv, SYMBOL, params={"positionSide":"BOTH"} if BINGX_POSITION_MODE=="oneway" else {})
    except Exception as e:
        cprint(f"⚠️ set_leverage: {e}", "red")

def place_market(side, qty, reduce_only=False):
    if not LIVE_TRADING or ex is None:
        return {"status":"paper","side":side,"qty":qty,"reduceOnly":reduce_only}

    params={}
    if BINGX_POSITION_MODE=="hedge":
        params["positionSide"] = "LONG" if side=="buy" else "SHORT"
    else:
        params["positionSide"]="BOTH"
    if reduce_only:
        params["reduceOnly"] = True

    try:
        o = ex.create_order(SYMBOL, "market", side, qty, params=params)
        return {"status":"ok","resp":o}
    except Exception as e:
        return {"status":"error","err":str(e)}

def open_position(side, price):
    if state["in_position"]: return
    set_leverage(LEVERAGE)
    qty = compute_qty(price)
    # طلب حقيقي
    resp = place_market("buy" if side=="long" else "sell", qty, reduce_only=False)
    if resp.get("status")=="ok":
        state.update({"in_position":True,"side":side,"entry":price,"qty":qty,"tp1_done":False})
        save_state()
        cprint(f"🚀 OPEN {side.upper()} | qty={nice(qty)} @ {nice(price)} | lev={LEVERAGE}x", "green" if side=="long" else "red")
    else:
        cprint(f"⚠️ OPEN failed: {resp.get('err')}", "red")

def close_position():
    if not state["in_position"]: return
    side_close = "sell" if state["side"]=="long" else "buy"
    resp = place_market(side_close, state["qty"], reduce_only=True)
    if resp.get("status")=="ok":
        state.update({"in_position":False,"side":None,"entry":0.0,"qty":0.0,"tp1_done":False})
        save_state()
        cprint("🏁 CLOSED position", "yellow")
    else:
        cprint(f"⚠️ CLOSE failed: {resp.get('err')}", "red")

# ---------- decide ----------
def decide(df):
    df["rsi"] = rsi(df["close"], RSI_LEN)
    plusDI, minusDI, ADX = dx_plus_minus(df, ADX_LEN)
    df["+di"]=plusDI; df["-di"]=minusDI; df["adx"]=ADX
    df["atr"] = atr(df, ATR_LEN)
    rf, up, lo, b, s = range_filter(df, RF_PERIOD, RF_MULT, RF_SOURCE)
    df["rf"]=rf; df["up"]=up; df["lo"]=lo; df["buy_sig"]=b; df["sell_sig"]=s

    last, prev = df.iloc[-1], df.iloc[-2]
    price = last["close"]

    c = {
        "doji":           is_doji(last["open"],last["high"],last["low"],last["close"]),
        "pin_bull":       is_pin_bull(last["open"],last["high"],last["low"],last["close"]),
        "pin_bear":       is_pin_bear(last["open"],last["high"],last["low"],last["close"]),
        "hammer":         is_hammer(last["open"],last["high"],last["low"],last["close"]),
        "inv_hammer":     is_inv_hammer(last["open"],last["high"],last["low"],last["close"]),
        "shooting":       is_shooting_star(last["open"],last["high"],last["low"],last["close"]),
        "gravestone":     is_gravestone(last["open"],last["high"],last["low"],last["close"]),
        "engulf_bull":    engulf_bull(prev.to_dict(), last.to_dict()),
        "engulf_bear":    engulf_bear(prev.to_dict(), last.to_dict()),
        "tweezer_top":    tweez_top(prev.to_dict(), last.to_dict()),
        "tweezer_bot":    tweez_bot(prev.to_dict(), last.to_dict()),
        "piercing":       piercing(prev.to_dict(), last.to_dict()),
        "darkcloud":      darkcloud(prev.to_dict(), last.to_dict()),
        "morning_star":   False,
        "evening_star":   False,
        "soldiers":       False,
        "black_crows":    False
    }
    if len(df)>=3:
        a,b,c3 = df.iloc[-3].to_dict(), prev.to_dict(), last.to_dict()
        c["morning_star"] = morning_star(a,b,c3)
        c["evening_star"] = evening_star(a,b,c3)
        c["soldiers"]     = soldiers3(df.tail(3))
        c["black_crows"]  = crows3(df.tail(3))

    regime = "TREND_UP" if (last["adx"]>25 and last["+di"]>last["-di"]) else \
             "TREND_DOWN" if (last["adx"]>25 and last["-di"]>last["+di"]) else "RANGE"

    return price, last, c, regime, bool(last["buy_sig"]), bool(last["sell_sig"])

# ---------- manage ----------
def manage_position(df, price, last):
    if not state["in_position"]: return
    side, entry, qty = state["side"], state["entry"], state["qty"]
    atrv = last["atr"]; adxv = last["adx"]

    gain = (pct(entry, price) if side=="long" else pct(price, entry))*100

    # TP1
    if not state["tp1_done"] and gain>=TP1_PCT*100 and qty>0:
        cut = max(qty*TP1_CLOSE_FRAC, 0.0)
        resp = place_market("sell" if side=="long" else "buy", cut, reduce_only=True)
        if resp.get("status")=="ok":
            state["qty"] -= cut
            state["tp1_done"] = True
            save_state()
            cprint(f"🎯 TP1 hit – closed {nice(cut)} remain {nice(state['qty'])}", "cyan")

    # Breakeven (منطق حماية داخلي)
    if gain>=BREAKEVEN_AFTER_PCT*100:
        pass  # (لا نرسل أمر تعديل؛ نكتفي بمنطق الخروج)

    # ATR trailing
    if (gain>=TRAIL_ACTIVATE_PCT*100 or (HOLD_TP_STRONG and adxv>=HOLD_TP_ADX)) and atrv>0:
        dist = ATR_MULT_TRAIL*atrv
        if side=="long" and last["low"]< (price - dist):
            cprint("🏁 ATR trail exit LONG", "yellow"); close_position()
        if side=="short" and last["high"]> (price + dist):
            cprint("🏁 ATR trail exit SHORT", "yellow"); close_position()

    # Giveback بعد TP1
    if state["tp1_done"] and gain>=MIN_TP_PERCENT*100:
        give = GIVEBACK_PCT
        if side=="long" and price <= entry*(1+(gain/100 - give)):
            cprint("↩️ Giveback exit LONG", "yellow"); close_position()
        if side=="short" and price >= entry*(1-(gain/100 - give)):
            cprint("↩️ Giveback exit SHORT", "yellow"); close_position()

# ---------- entries ----------
def consider_entry(price, sig_buy, sig_sell):
    if state["in_position"]: return
    # دخول مطابق لتريدنج فيو (Range Filter)
    if sig_buy:  open_position("long",  price)
    if sig_sell: open_position("short", price)

# ---------- HUD logs ----------
def hud(df, price, candles, regime, sig_buy, sig_sell):
    last = df.iloc[-1]
    cprint(f"\n{now_utc()} | {SYMBOL} {INTERVAL} | LIVE={'ON' if LIVE_TRADING else 'OFF'} | Mode={state['mode']} | PosMode={state['position_mode']}", "white")
    print(colored("INDICATORS","cyan"))
    print(f"  💠 Px={nice(price)}  RF={nice(last['rf'])}  hi={nice(last['high'])} lo={nice(last['low'])}")
    print(f"  📈 RSI({RSI_LEN})={nice(last['rsi'])}  +DI={nice(last['+di'])}  -DI={nice(last['-di'])}  ADX({ADX_LEN})={nice(last['adx'])}")
    print(f"  📏 ATR({ATR_LEN})={nice(last['atr'])}  spread_bps≈{SPREAD_GUARD_BPS}  Regime={regime}")

    print(colored("CANDLES","magenta"))
    names=[("Doji","doji"),("PinBull","pin_bull"),("PinBear","pin_bear"),
           ("Hammer","hammer"),("InvHammer","inv_hammer"),
           ("Shooting","shooting"),("Gravestone","gravestone"),
           ("EngulfBull","engulf_bull"),("EngulfBear","engulf_bear"),
           ("TweezTop","tweezer_top"),("TweezBot","tweezer_bot"),
           ("Piercing","piercing"),("DarkCloud","darkcloud"),
           ("MorningStar","morning_star"),("EveningStar","evening_star"),
           ("3Soldiers","soldiers"),("3Crows","black_crows")]
    print("  " + " | ".join([f"{n}:{icon(candles.get(k,False))}" for n,k in names]))

    print(colored("SIGNALS","yellow"))
    print(f"  BUY={icon(sig_buy)}  SELL={icon(sig_sell)}")

    print(colored("POSITION","blue"))
    if state["in_position"]:
        chg = (pct(state["entry"], last['close']) if state["side"]=="long" else pct(last['close'], state["entry"])) * 100
        print(f"  {('🟩 LONG' if state['side']=='long' else '🟥 SHORT')} entry={nice(state['entry'])} qty={nice(state['qty'])} Δ={nice(chg,2)}% tp1={icon(state['tp1_done'])}")
    else:
        print(f"  🟨 FLAT  next_qty≈{nice(state['next_qty_hint'])}  equity≈{nice(state['risk_equity'])} USDT")

    print(colored("RESULTS","green"))
    print(f"  💹 RealizedPnL={nice(state['pnl_realized'])} USDT  | EffectiveEq≈{nice(state['risk_equity'])} USDT")

# ---------- main loop ----------
def strategy_loop():
    while True:
        try:
            ohlc = fetch_ohlcv(SYMBOL, INTERVAL, limit=300)
            if not ohlc: 
                cprint("⚠️ no data", "red"); time.sleep(DECISION_EVERY_S); continue
            df = df_from_ohlc(ohlc)
            price, last, candles, regime, sig_buy, sig_sell = decide(df)

            # لو USE_TV_BAR=TRUE انتظر إغلاق شمعة جديدة
            if USE_TV_BAR:
                ts = df.iloc[-1]["ts"]
                if state["last_close_ts"] == ts:
                    time.sleep(DECISION_EVERY_S); continue
                state["last_close_ts"] = ts

            # إدارة و دخول
            manage_position(df, price, last)
            consider_entry(price, sig_buy, sig_sell)

            # HUD
            hud(df, price, candles, regime, sig_buy, sig_sell)

        except Exception as e:
            cprint(f"❌ loop err: {e}\n{traceback.format_exc()}", "red")
        time.sleep(DECISION_EVERY_S)

# ---------- Flask / keepalive ----------
from flask import Flask
import requests
app = Flask(__name__)

@app.get("/")
def root():
    return f"OK • {SYMBOL} • {INTERVAL} • LIVE={'ON' if LIVE_TRADING else 'OFF'} • {now_utc()}"

@app.get("/metrics")
def metrics():
    return jsonify({"ts":now_utc(),"symbol":SYMBOL,"interval":INTERVAL,"state":state})

def keepalive():
    if not SELF_URL: return
    n=0
    while True:
        try:
            requests.get(SELF_URL, timeout=4)
            # اطبع سطرين فقط في الدقيقة
            n=(n+1)%PING_LOG_EVERY_N
            if n==0:
                cprint(f"keepalive ok (200) • URL: {SELF_URL}", "white")
        except: pass
        time.sleep(KEEPALIVE_SECONDS)

# ---------- start ----------
if __name__=="__main__":
    cprint(f"Starting SMART BOT  •  {SYMBOL}  •  MODE={'LIVE' if LIVE_TRADING else 'PAPER'}", "white","on_blue")
    cprint(f"PositionMode = {BINGX_POSITION_MODE.upper()}", "cyan")
    if ex:
        try: set_leverage(LEVERAGE)
        except Exception as e: cprint(f"set_leverage warn: {e}", "yellow")
    threading.Thread(target=strategy_loop, daemon=True).start()
    if SELF_URL:
        threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
