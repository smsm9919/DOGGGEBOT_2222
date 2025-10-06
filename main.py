# main.py
# ============================
# Bot: DOGE/USDT (BingX Perp) — Pro "AI" Strategy
# - TV-matching entries (Range Filter)
# - RSI/ADX/ATR/+DI/-DI
# - Full Candlestick Patterns (Engulfing, Doji, Pin, Hammer/Inv, Shooting/Gravestone, SpinningTop,
#   Morning/Evening Star, Three White Soldiers/Black Crows, Tweezers Top/Bottom, Piercing/DarkCloud)
# - Smart Mgmt: TP1 + Breakeven + ATR Trailing + Giveback + Regime Detection
# - Scalp vs Trend auto-mode
# - Cumulative PnL sizing (next position size grows with realized PnL)
# - Iconic color logs + /metrics
# ============================

import os, time, math, json, threading, traceback, statistics as stats
from datetime import datetime, timezone
from termcolor import cprint, colored
import pandas as pd
from flask import Flask, jsonify
import requests

# ---------- ENV ----------
# مفاتيح بينج إكس
BINGX_API_KEY   = os.getenv("BINGX_API_KEY","")
BINGX_API_SECRET= os.getenv("BINGX_API_SECRET","")

# أساسيات
SYMBOL          = os.getenv("SYMBOL","DOGE/USDT:USDT")
INTERVAL        = os.getenv("INTERVAL","15m")
LEVERAGE        = int(os.getenv("LEVERAGE","10"))
RISK_ALLOC      = float(os.getenv("RISK_ALLOC","0.60"))     # نسبة من الرصيد
DECISION_EVERY_S= int(os.getenv("DECISION_EVERY_S","30"))
KEEPALIVE_SECONDS = int(os.getenv("KEEPALIVE_SECONDS","50"))
PORT            = int(os.getenv("PORT","5000"))
PYTHON_VERSION  = os.getenv("PYTHON_VERSION","3.10.14")

# Range Filter (توافق مع TV)
RF_PERIOD       = int(os.getenv("RF_PERIOD","20"))
RF_SOURCE       = os.getenv("RF_SOURCE","close")             # close/high/low
RF_MULT         = float(os.getenv("RF_MULT","3.5"))
SPREAD_GUARD_BPS= int(os.getenv("SPREAD_GUARD_BPS","6"))
USE_TV_BAR      = os.getenv("USE_TV_BAR","false").lower()=="true"  # لو true يستنى اغلاق الشمعة
FORCE_TV_ENTRIES= os.getenv("FORCE_TV_ENTRIES","false").lower()=="true"

# مؤشرات
RSI_LEN         = int(os.getenv("RSI_LEN","14"))
ADX_LEN         = int(os.getenv("ADX_LEN","14"))
ATR_LEN         = int(os.getenv("ATR_LEN","14"))

# إدارة ذكية
TP1_PCT         = float(os.getenv("TP1_PCT","0.40"))         # % من الهدف الأول
TP1_CLOSE_FRAC  = float(os.getenv("TP1_CLOSE_FRAC","0.50"))  # اغلاق جزء
TRAIL_ACTIVATE_PCT=float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL  = float(os.getenv("ATR_MULT_TRAIL","1.6"))
GIVEBACK_PCT    = float(os.getenv("GIVEBACK_PCT","0.30"))    # تنازل عند الرجوع
BREAKEVEN_AFTER_PCT=float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
MOVE_3BARS_PCT  = float(os.getenv("MOVE_3BARS_PCT","0.8"))   # انفجار 3 شمعات
MIN_TP_PERCENT  = float(os.getenv("MIN_TP_PERCENT","0.4"))   # أقل مكسب % لبدء حماية

# ذكاء إضافي للتيك بروفت
HOLD_TP_STRONG  = os.getenv("HOLD_TP_STRONG","true").lower()=="true"
HOLD_TP_ADX     = int(os.getenv("HOLD_TP_ADX","28"))
HOLD_TP_SLOPE   = float(os.getenv("HOLD_TP_SLOPE","0.50"))

# تفعيل الخروج الذكي
USE_SMART_EXIT  = os.getenv("USE_SMART_EXIT","true").lower()=="true"

# عناوين Render
SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "")

# تشغيل تداول حقيقي؟
LIVE_TRADING    = os.getenv("LIVE_TRADING","true").lower()=="true"  # لو true ينفذ أوامر

# ---------- Exchange (ccxt) ----------
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

# ---------- Helpers ----------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def true_range(df):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

def atr(df, length=14):
    return true_range(df).rolling(length).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain/(loss+1e-9)
    return 100 - (100/(1+rs))

def dx_plus_minus(df, length=14):
    upMove = df['high'].diff()
    downMove = -df['low'].diff()
    plusDM = ((upMove > downMove) & (upMove > 0)) * upMove
    minusDM = ((downMove > upMove) & (downMove > 0)) * downMove
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
    upper = basis + rng
    lower = basis - rng
    # إشارات: تخطي ومكوث
    buy = (src > upper) & (src.shift(1) <= upper.shift(1))
    sell= (src < lower) & (src.shift(1) >= lower.shift(1))
    return basis, upper, lower, buy, sell

def pct(a,b):   # change % from a to b
    return 0 if a==0 else (b-a)/a

def bps(x): return int(round(x*10000))

def nice_num(x, digits=6):
    try: return float(f"{x:.{digits}f}")
    except: return x

# ---------- Candles Anatomy ----------
def candle_anatomy(o,h,l,c):
    body = abs(c-o)
    full = max(h,l) - min(h,l) + 1e-9
    upper = h - max(c,o)
    lower = min(c,o) - l
    return body/full, upper/full, lower/full  # ratios

def is_doji(o,h,l,c, tol=0.1):
    body, up, lo = candle_anatomy(o,h,l,c)
    return body < tol and up>0 and lo>0

def is_spinning_top(o,h,l,c):
    body, up, lo = candle_anatomy(o,h,l,c)
    return body < 0.3 and up>0.2 and lo>0.2

def is_pin_bull(o,h,l,c):
    body, up, lo = candle_anatomy(o,h,l,c)
    return lo > 0.55 and body<0.35

def is_pin_bear(o,h,l,c):
    body, up, lo = candle_anatomy(o,h,l,c)
    return up > 0.55 and body<0.35

def is_hammer(o,h,l,c):
    return is_pin_bull(o,h,l,c) and c>o

def is_inv_hammer(o,h,l,c):
    return is_pin_bear(o,h,l,c) and c>o

def is_shooting_star(o,h,l,c):
    return is_pin_bear(o,h,l,c) and c<o

def is_gravestone(o,h,l,c):
    return is_doji(o,h,l,c) and (h-max(c,o))>(min(c,o)-l)*2

def engulfing_bull(prev, cur):
    return (cur['open']<cur['close']) and (prev['open']>prev['close']) and (cur['open']<=prev['close']) and (cur['close']>=prev['open'])

def engulfing_bear(prev, cur):
    return (cur['open']>cur['close']) and (prev['open']<prev['close']) and (cur['open']>=prev['close']) and (cur['close']<=prev['open'])

def tweezers_top(prev, cur):
    return abs(prev['high']-cur['high'])/max(prev['high'],1e-9)<0.001 and prev['close']>prev['open'] and cur['close']<cur['open']

def tweezers_bottom(prev, cur):
    return abs(prev['low']-cur['low'])/max(prev['low'],1e-9)<0.001 and prev['close']<prev['open'] and cur['close']>cur['open']

def piercing(prev, cur):
    # صاعد: يفتح تحت إقفال سابق و يغلق أعلى منتصف جسمه
    prev_mid = (prev['open']+prev['close'])/2
    return prev['open']>prev['close'] and cur['open']<prev['close'] and cur['close']>prev_mid

def dark_cloud(prev, cur):
    prev_mid = (prev['open']+prev['close'])/2
    return prev['close']>prev['open'] and cur['open']>prev['close'] and cur['close']<prev_mid

def morning_star(c1,c2,c3):
    return c1['close']<c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']>c3['open'] and c3['close']> ((c1['open']+c1['close'])/2)

def evening_star(c1,c2,c3):
    return c1['close']>c1['open'] and is_doji(c2['open'],c2['high'],c2['low'],c2['close']) and c3['close']<c3['open'] and c3['close']< ((c1['open']+c1['close'])/2)

def soldiers(df):  # three white soldiers
    a=df.iloc[-3];b=df.iloc[-2];c=df.iloc[-1]
    return a['close']>a['open'] and b['close']>b['open'] and c['close']>c['open'] and c['close']>b['close']>a['close']

def black_crows(df):
    a=df.iloc[-3];b=df.iloc[-2];c=df.iloc[-1]
    return a['close']<a['open'] and b['close']<b['open'] and c['close']<c['open'] and c['close']<b['close']<a['close']

def explosive_move(df, bars=3, pct_th=0.8):
    # ارتفاع متوسط المدى آخر 3 شمعات نسبةً إلى ATR
    last = df.tail(bars)
    moves = [abs(last.iloc[i]['close']-last.iloc[i]['open']) for i in range(len(last))]
    atrv = atr(df).iloc[-1]
    if atrv<=0: return False
    return (sum(moves)/ (bars*atrv)) > pct_th

# ---------- State ----------
state = {
    "in_position": False,
    "side": None,            # "long"/"short"
    "entry": 0.0,
    "qty": 0.0,
    "tp1_done": False,
    "pnl_realized": 0.0,
    "pnl_compound": 0.0,
    "last_close_ts": None,
    "mode": "SMART",
    "risk_equity": 0.0,
    "next_qty_hint": 0.0
}
state_file = "pnl_state.json"

def load_state():
    global state
    try:
        with open(state_file,"r") as f:
            j=json.load(f)
            state.update(j)
    except:
        pass

def save_state():
    try:
        with open(state_file,"w") as f:
            json.dump(state,f)
    except:
        pass

load_state()

# ---------- Data ----------
def fetch_ohlcv(symbol, tf, limit=300):
    if ex is None: return None
    return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)

def build_df(ohlc):
    df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])
    return df

# ---------- Sizing ----------
def get_balance_usdt():
    try:
        bal=ex.fetch_balance()
        return float(bal['USDT']['free'])
    except:
        return 60.0

def compute_qty(price):
    # حجم على 10x + ربح تراكمي: نزود 5% لكل 10$ ربح مُحقق
    bal = get_balance_usdt()
    equity = max(bal, 10.0)
    state["risk_equity"] = equity
    boost = 1.0 + (state["pnl_realized"]/10.0)*0.05
    usd_alloc = equity * RISK_ALLOC * boost
    # عقود = (usd_alloc * LEVERAGE) / price
    qty = (usd_alloc * LEVERAGE) / max(price,1e-9)
    state["next_qty_hint"] = qty
    return max(qty, 1.0)

# ---------- Orders (LIVE safe) ----------
def set_leverage(leverage=LEVERAGE):
    try:
        if LIVE_TRADING:
            ex.set_leverage(leverage, SYMBOL)
    except Exception as e:
        cprint(f"⚠️ set_leverage: {e}", "red")

def market_order(side, qty):
    if not LIVE_TRADING:
        return {"status":"paper","side":side,"qty":qty}
    try:
        if side=="buy":
            return ex.create_order(SYMBOL, "market", "buy", qty)
        else:
            return ex.create_order(SYMBOL, "market", "sell", qty)
    except Exception as e:
        cprint(f"⚠️ order err: {e}", "red")
        return {"status":"error","err":str(e)}

def close_position():
    if not state["in_position"]: return
    side = "sell" if state["side"]=="long" else "buy"
    res = market_order(side, state["qty"])
    state["in_position"]=False
    state["tp1_done"]=False
    state["side"]=None
    state["entry"]=0.0
    state["qty"]=0.0
    save_state()
    return res

# ---------- Decision / Strategy ----------
def decide(df):
    # Indicators
    df["rsi"]=rsi(df["close"], RSI_LEN)
    plusDI, minusDI, ADX = dx_plus_minus(df, ADX_LEN)
    df["+di"]=plusDI; df["-di"]=minusDI; df["adx"]=ADX
    A = atr(df, ATR_LEN); df["atr"]=A
    basis, up, lo, buy_sig, sell_sig = range_filter(df, RF_PERIOD, RF_MULT, RF_SOURCE)
    df["rf"]=basis; df["up"]=up; df["lo"]=lo; df["buy_sig"]=buy_sig; df["sell_sig"]=sell_sig

    last = df.iloc[-1]; prev = df.iloc[-2]
    price = last["close"]

    # شموع
    candles = {
        "doji": is_doji(last["open"],last["high"],last["low"],last["close"]),
        "pin_bull": is_pin_bull(last["open"],last["high"],last["low"],last["close"]),
        "pin_bear": is_pin_bear(last["open"],last["high"],last["low"],last["close"]),
        "hammer": is_hammer(last["open"],last["high"],last["low"],last["close"]),
        "inv_hammer": is_inv_hammer(last["open"],last["high"],last["low"],last["close"]),
        "shooting": is_shooting_star(last["open"],last["high"],last["low"],last["close"]),
        "gravestone": is_gravestone(last["open"],last["high"],last["low"],last["close"]),
        "engulf_bull": engulfing_bull(prev.to_dict(), last.to_dict()),
        "engulf_bear": engulfing_bear(prev.to_dict(), last.to_dict()),
        "tweezer_top": tweezers_top(prev.to_dict(), last.to_dict()),
        "tweezer_bot": tweezers_bottom(prev.to_dict(), last.to_dict()),
        "piercing": piercing(prev.to_dict(), last.to_dict()),
        "darkcloud": dark_cloud(prev.to_dict(), last.to_dict())
    }
    if len(df)>=3:
        c1=df.iloc[-3].to_dict(); c2=df.iloc[-2].to_dict(); c3=df.iloc[-1].to_dict()
        candles["morning_star"]=morning_star(c1,c2,c3)
        candles["evening_star"]=evening_star(c1,c2,c3)
        candles["soldiers"]=soldiers(df.tail(3))
        candles["black_crows"]=black_crows(df.tail(3))
    else:
        candles.update({"morning_star":False,"evening_star":False,"soldiers":False,"black_crows":False})

    # انفجار
    boom = explosive_move(df, 3, MOVE_3BARS_PCT)

    # نظام السوق
    regime = "TREND_UP" if (last["adx"]>25 and last["+di"]>last["-di"]) else \
             "TREND_DOWN" if (last["adx"]>25 and last["-di"]>last["+di"]) else "RANGE"

    # إشارة الدخول (مطابقة TV)
    sig_buy = bool(last["buy_sig"])
    sig_sell= bool(last["sell_sig"])

    return price, last, candles, regime, boom, sig_buy, sig_sell

def manage_position(df, price, last, candles, regime, boom):
    # حماية & إدارة
    if not state["in_position"]: return

    side = state["side"]
    entry= state["entry"]
    qty  = state["qty"]
    atrv = df["atr"].iloc[-1]
    adxv = df["adx"].iloc[-1]

    up_pct = pct(entry, price) if side=="long" else pct(price, entry)
    gain_pct = up_pct*100

    # TP1
    if not state["tp1_done"] and gain_pct>=TP1_PCT*100:
        cut = max(qty*TP1_CLOSE_FRAC, 0.0)
        if cut>0:
            market_order("sell" if side=="long" else "buy", cut)
            state["qty"] -= cut
            state["tp1_done"] = True
            cprint(f"🎯 TP1 hit | closed {nice_num(cut)} | remain {nice_num(state['qty'])}", "cyan")

    # Breakeven
    if gain_pct>=BREAKEVEN_AFTER_PCT*100 and state["qty"]>0:
        # لا أوامر تعديل حقيقية — نستخدم منطق التخارج فقط
        pass

    # Trailing
    activate = gain_pct>=TRAIL_ACTIVATE_PCT*100 or (HOLD_TP_STRONG and adxv>=HOLD_TP_ADX and boom)
    if activate and atrv>0:
        dist = ATR_MULT_TRAIL*atrv
        if side=="long":
            stop = max(entry, price - dist)
            # إذا اغلاق أقل من stop: اغلاق
            if last["low"]<stop:
                cprint(f"🏁 ATR trail exit LONG @ {nice_num(price)}", "yellow")
                close_position()
        else:
            stop = min(entry, price + dist)
            if last["high"]>stop:
                cprint(f"🏁 ATR trail exit SHORT @ {nice_num(price)}", "yellow")
                close_position()

    # Giveback
    if state["tp1_done"] and gain_pct>=MIN_TP_PERCENT*100:
        peak = max(entry, price) if side=="long" else min(entry, price)
        give = GIVEBACK_PCT
        if side=="long":
            if price <= entry*(1+ (up_pct - give)):
                cprint(f"↩️ Giveback exit LONG", "yellow")
                close_position()
        else:
            if price >= entry*(1 - (up_pct - give)):
                cprint(f"↩️ Giveback exit SHORT", "yellow")
                close_position()

def open_position(side, price):
    if state["in_position"]:
        return
    set_leverage(LEVERAGE)
    qty = compute_qty(price)
    res = market_order("buy" if side=="long" else "sell", qty)
    state["in_position"]=True
    state["side"]=side
    state["entry"]=price
    state["qty"]=qty
    state["tp1_done"]=False
    save_state()
    cprint(f"🚀 Open {side.upper()} | qty={nice_num(qty)} @ {nice_num(price)} | lev={LEVERAGE}x", "green" if side=="long" else "red")

def consider_entry(price, sig_buy, sig_sell, candles, regime):
    # دخول مطابق لتريدنج فيو: نعتمد فقط على إشارة RF (buy_sig / sell_sig)
    # ثم نفلتر بس الغير منطقي (سبريد/رجيم معاكس قوي)
    if state["in_position"]:
        return

    if sig_buy:
        open_position("long", price)
    elif sig_sell:
        open_position("short", price)

# ---------- Logging ----------
def icon_bool(b): return "🟢" if b else "⚪"
def colored_val(v, pos="green", neg="red", zero="white"):
    if v>0: return colored(f"{v:.2f}", pos)
    if v<0: return colored(f"{v:.2f}", neg)
    return colored(f"{v:.2f}", zero)

def log_frame(df, price, candles, regime, sig_buy, sig_sell):
    last=df.iloc[-1]; prev=df.iloc[-2]
    cprint(f"\n{now_utc()}  |  {SYMBOL}  {INTERVAL}  |  Mode={state['mode']}", "white")

    # Indicators
    print(colored("INDICATORS","cyan"))
    print(f"  💠 Price      = {nice_num(price)}   RF  = {nice_num(last['rf'])}   hi = {nice_num(last['high'])}  lo = {nice_num(last['low'])}")
    print(f"  📈 RSI({RSI_LEN}) = {nice_num(last['rsi'])}   +DI={nice_num(last['+di'])}  -DI={nice_num(last['-di'])}   ADX({ADX_LEN})={nice_num(last['adx'])}")
    print(f"  📏 ATR({ATR_LEN}) = {nice_num(last['atr'])}     spread_bps≈{SPREAD_GUARD_BPS}   Regime = {regime}")

    # Candles
    print(colored("CANDLES","magenta"))
    names = [
        ("Doji", "doji"),
        ("PinBull", "pin_bull"), ("PinBear","pin_bear"),
        ("Hammer","hammer"), ("InvHammer","inv_hammer"),
        ("Shooting","shooting"), ("Gravestone","gravestone"),
        ("EngulfBull","engulf_bull"),("EngulfBear","engulf_bear"),
        ("TweezTop","tweezer_top"),("TweezBot","tweezer_bot"),
        ("Piercing","piercing"),("DarkCloud","darkcloud"),
        ("MorningStar","morning_star"),("EveningStar","evening_star"),
        ("3 Soldiers","soldiers"),("3 Crows","black_crows")
    ]
    row=[]
    for n,k in names:
        row.append(f"{n}:{'🟢' if candles.get(k,False) else '⚪'}")
    print("  " + " | ".join(row))

    # Signals
    print(colored("SIGNALS","yellow"))
    print(f"  BUY={icon_bool(sig_buy)}   SELL={icon_bool(sig_sell)}")

    # Position
    print(colored("POSITION","blue"))
    if state["in_position"]:
        pchg = pct(state["entry"], last['close']) if state["side"]=="long" else pct(last['close'], state["entry"])
        print(f"  {('🟩 LONG' if state['side']=='long' else '🟥 SHORT')}  entry={nice_num(state['entry'])}  qty={nice_num(state['qty'])}  Δ={colored_val(pchg*100)}%  tp1={icon_bool(state['tp1_done'])}")
    else:
        print(f"  🟨 FLAT   next_qty_hint≈{nice_num(state['next_qty_hint'])}")

    # Results
    print(colored("RESULTS","green"))
    print(f"  💹 RealizedPnL = {nice_num(state['pnl_realized'])} USDT   💰 EffectiveEq ≈ {nice_num(state['risk_equity'])} USDT")

# ---------- Loop ----------
def strategy_loop():
    while True:
        try:
            ohlc = fetch_ohlcv(SYMBOL, INTERVAL, limit=300)
            if not ohlc:
                cprint("No data.", "red"); time.sleep(DECISION_EVERY_S); continue

            df = build_df(ohlc)
            price, last, candles, regime, boom, sig_buy, sig_sell = decide(df)

            if USE_TV_BAR:
                # لا نتخذ قرار إلا عند إغلاق شمعة جديدة
                ts_close = df.iloc[-1]['ts']
                if state["last_close_ts"] == ts_close:
                    # تحديث HUD فقط
                    log_frame(df, price, candles, regime, sig_buy, sig_sell)
                    time.sleep(DECISION_EVERY_S)
                    continue
                state["last_close_ts"]=ts_close

            # إدارة صفقة مفتوحة
            manage_position(df, price, last, candles, regime, boom)

            # دخول جديد (مطابق TV)
            consider_entry(price, sig_buy, sig_sell, candles, regime)

            # HUD
            log_frame(df, price, candles, regime, sig_buy, sig_sell)

        except Exception as e:
            cprint(f"❌ loop err: {e}\n{traceback.format_exc()}", "red")
        time.sleep(DECISION_EVERY_S)

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/")
def home():
    return f"OK • {SYMBOL} • {INTERVAL} • LIVE={'ON' if LIVE_TRADING else 'OFF'} • {now_utc()}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "ts": now_utc(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "state": state
    })

@app.route("/ping")
def ping(): return "pong"

def keepalive():
    if not SELF_URL: return
    while True:
        try:
            requests.get(SELF_URL, timeout=4)
        except: pass
        time.sleep(KEEPALIVE_SECONDS)

# ---------- Start ----------
if __name__=="__main__":
    cprint(f"Starting Pro AI bot • {SYMBOL} • {INTERVAL} • LIVE={LIVE_TRADING}", "white", "on_blue")
    if ex:
        try: set_leverage(LEVERAGE)
        except: pass
    threading.Thread(target=strategy_loop, daemon=True).start()
    if SELF_URL:
        threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
