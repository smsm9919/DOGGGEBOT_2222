# -*- coding: utf-8 -*-
"""
Bots: DOGE/USDT (BingX Perp)
- Entry: EXACTLY from TradingView webhook (BUY/SELL)
- Post-entry Intelligence: Candles + RSI/ADX/DX/ATR + Regime/Range
- Smart PnL: TP1 partial + Breakeven + ATR Trailing
- Pro logs with icons
- Flask keepalive + /tv webhook + /metrics
"""

import os, time, json, math, threading, traceback
from datetime import datetime, timezone
from termcolor import colored
import requests
import pandas as pd

# ====== ccxt (BingX Perp) ======
import ccxt

# ---------------------- ENV ----------------------
ENV = lambda k, d=None: os.getenv(k, d)

SYMBOL              = ENV("SYMBOL", "DOGE/USDT:USDT")
INTERVAL            = ENV("INTERVAL", "15m")
LEVERAGE            = int(ENV("LEVERAGE", "10"))
RISK_ALLOC          = float(ENV("RISK_ALLOC", "0.60"))       # % من الرصيد يدخل به
SPREAD_GUARD_BPS    = int(ENV("SPREAD_GUARD_BPS", "6"))      # حماية من السبريد
DECISION_EVERY_S    = int(ENV("DECISION_EVERY_S", "30"))
KEEPALIVE_SECONDS   = int(ENV("KEEPALIVE_SECONDS", "50"))
PORT                = int(ENV("PORT", "5000"))

# مؤشرات
RSI_LEN             = int(ENV("RSI_LEN", "14"))
ADX_LEN             = int(ENV("ADX_LEN", "14"))
ATR_LEN             = int(ENV("ATR_LEN", "14"))

# ربح ذكي
TP1_PCT             = float(ENV("TP1_PCT", "0.40"))          # 40% من الكمية
TP1_CLOSE_FRAC      = float(ENV("TP1_CLOSE_FRAC", "0.50"))   # إغلاق 50% في TP1
TRAIL_ACTIVATE_PCT  = float(ENV("TRAIL_ACTIVATE_PCT", "0.60"))
ATR_MULT_TRAIL      = float(ENV("ATR_MULT_TRAIL", "1.6"))
BREAKEVEN_AFTER_PCT = float(ENV("BREAKEVEN_AFTER_PCT", "0.30"))
COOLDOWN_AFTER_CLOSE_BARS = int(ENV("COOLDOWN_AFTER_CLOSE_BARS", "0"))

# TV
FORCE_TV_ENTRIES    = ENV("FORCE_TV_ENTRIES", "true").lower() == "true"
USE_TV_BAR          = ENV("USE_TV_BAR", "false").lower() == "true"  # لو true يشتغل على إغلاق الشمعة
RENDER_EXTERNAL_URL = ENV("RENDER_EXTERNAL_URL", "")

# مفاتيح بينج إكس
API_KEY             = ENV("BINGX_API_KEY", "")
API_SECRET          = ENV("BINGX_API_SECRET", "")

# طباعة Unicode
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------- Icons ----------
I = {
    "rf":"🧰","rsi":"📈","adx":"📊","dx":"🧭","atr":"📐","c":"🕯️",
    "buy":"🟢","sell":"🔴","flat":"⚪","wait":"🟡","ok":"✅","no":"❌",
    "up":"📈","dn":"📉","eng":"🧱","doji":"➕","pin":"📌","star":"⭐",
    "server":"🌐","wallet":"👛","pos":"🎯","trail":"🪢","be":"🛡️","tp":"🎯",
}

# ---------------------- STATE ----------------------
class State:
    def __init__(self):
        self.last_tv = None         # {"side":"BUY"/"SELL","price":float,"ts":epoch,"bar_ts":iso?}
        self.last_tv_at = 0
        self.pos = None             # dict when open
        self.cooldown_bars = 0
        self.compound_pnl = 0.0
        self.mode = "SMART"
        self.effective_eq = 0.0
        self.last_keepalive = 0
        self.last_print_candle_close_in = None

S = State()

# ---------------------- EXCHANGE ----------------------
def make_bingx():
    ex = ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "marginMode": "isolated"
        }
    })
    return ex

EX = make_bingx()

def ensure_leverage():
    try:
        EX.set_leverage(LEVERAGE, SYMBOL)
    except Exception:
        pass

# ---------------------- OHLC / INDICATORS ----------------------
def fetch_ohlcv(limit=200):
    # ccxt timeframe mapping uses the same "15m"
    return EX.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1*delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

def adx_dx(df, length=14):
    h,l,c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = atr(df, 1)  # TR بدون متوسط
    tr_sm = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / tr_sm)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / tr_sm)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, dx, adx

# ---------------------- CANDLES INTELLIGENCE ----------------------
def candle_tags(df):
    """ترجع وسم/وسمين للشمعة الأخيرة (مثلاً ENGULF_BULL/BEAR, DOJI, PIN/SHOOTING/ HAMMER)"""
    row = df.iloc[-1]
    prev = df.iloc[-2]
    o,h,l,c = row["open"], row["high"], row["low"], row["close"]
    body = abs(c-o)
    full = h-l
    upper = h - max(o,c)
    lower = min(o,c) - l

    tags = []

    # Doji
    if full > 0 and body/full < 0.1:
        tags.append("DOJI")

    # Pin-like
    if full > 0:
        if upper/full > 0.6 and body/full < 0.3:
            tags.append("SHOOTING_STAR")
        if lower/full > 0.6 and body/full < 0.3:
            tags.append("HAMMER")

    # Engulfing
    po, pc = prev["open"], prev["close"]
    if (c > o) and (pc < po) and (c >= po) and (o <= pc):
        tags.append("ENGULF_BULL")
    if (c < o) and (pc > po) and (c <= po) and (o >= pc):
        tags.append("ENGULF_BEAR")

    return tags if tags else ["NONE"]

# ---------------------- REGIME ----------------------
def trend_regime(df):
    """بسيطة: slope EMA + ADX"""
    ema = df["close"].ewm(span=20, adjust=False).mean()
    slope = ema.diff().iloc[-1]
    _, _, _, adxv = adx_dx(df, ADX_LEN)
    adx_last = float(adxv.iloc[-1])
    if slope > 0 and adx_last >= 20:
        return "TREND_UP", adx_last
    if slope < 0 and adx_last >= 20:
        return "TREND_DOWN", adx_last
    return "NEUTRAL", adx_last

# ---------------------- POSITION & ORDERS ----------------------
def wallet_balance_usdt():
    try:
        bal = EX.fetch_balance(params={"type":"swap"})
        return float(bal["USDT"]["free"]) + float(bal["USDT"].get("used", 0))
    except Exception:
        return 0.0

def open_position(side, price, df):
    global S
    bal = wallet_balance_usdt()
    if bal <= 0:
        log(f"{I['wallet']} No balance", "red"); return False

    # احسب الكمية (isolated x Leverage)
    notional = bal * RISK_ALLOC * LEVERAGE
    qty = max(1.0, round(notional / price, 0))  # عقود تقريبية

    # حماية السبريد
    spread_bps = 0
    try:
        ob = EX.fetch_order_book(SYMBOL, 5)
        best_ask = ob["asks"][0][0]; best_bid = ob["bids"][0][0]
        spread_bps = (best_ask - best_bid)/((best_ask+best_bid)/2) * 10000
        if spread_bps > SPREAD_GUARD_BPS:
            log(f"Spread guard tripped ({spread_bps:.1f} bps) > {SPREAD_GUARD_BPS} bps", "yellow")
            return False
    except Exception:
        pass

    params = {"type":"market", "reduceOnly": False, "positionSide": "LONG" if side=="BUY" else "SHORT"}
    try:
        if side == "BUY":
            EX.create_market_buy_order(SYMBOL, qty, params=params)
        else:
            EX.create_market_sell_order(SYMBOL, qty, params=params)

        S.pos = {
            "side": side, "entry": price, "qty": qty,
            "tp1_done": False, "breakeven": None, "trail": None,
            "opened_at": utc_ts(),
        }
        log(f"{I['pos']} OPEN {side} @ {price:.6f}  qty={qty}  lev={LEVERAGE}x", "green" if side=="BUY" else "red")
        return True
    except Exception as e:
        log("order error: "+str(e), "red")
        return False

def close_partial(frac, reason=""):
    if not S.pos: return
    side = S.pos["side"]
    qty = S.pos["qty"] * frac
    qty = max(1.0, round(qty, 0))
    if qty <= 0: return
    params = {"type":"market", "reduceOnly": True,
              "positionSide": "LONG" if side=="BUY" else "SHORT"}
    try:
        if side == "BUY":   # نغلق بالبيع
            EX.create_market_sell_order(SYMBOL, qty, params=params)
        else:               # نغلق بالشراء
            EX.create_market_buy_order(SYMBOL, qty, params=params)
        S.pos["qty"] -= qty
        log(f"{I['tp']} Partial close {frac*100:.0f}% ({qty})  reason={reason}", "cyan")
        if S.pos["qty"] <= 0:
            S.pos = None
    except Exception as e:
        log("close error: "+str(e), "red")

def close_all(reason=""):
    if not S.pos: return
    close_partial(1.0, reason)

# ---------------------- SMART EXIT ENGINE ----------------------
def manage_after_entry(df):
    if not S.pos: return
    side = S.pos["side"]
    price = float(df["close"].iloc[-1])
    entry = S.pos["entry"]

    rr = (price - entry)/entry if side=="BUY" else (entry - price)/entry
    rr_pct = rr*100

    # ATR trailing
    atrv = float(atr(df, ATR_LEN).iloc[-1])

    # 1) TP1
    if not S.pos["tp1_done"] and rr_pct >= TP1_PCT*100:
        close_partial(TP1_CLOSE_FRAC, reason="TP1")
        S.pos["tp1_done"] = True

    # 2) Breakeven
    if S.pos["breakeven"] is None and rr_pct >= BREAKEVEN_AFTER_PCT*100:
        S.pos["breakeven"] = entry
        log(f"{I['be']} Breakeven armed @ {entry:.6f}", "yellow")

    # 3) ATR Trailing (يشتغل بعد التفعيل)
    if rr_pct >= TRAIL_ACTIVATE_PCT*100:
        if side=="BUY":
            trail_stop = price - ATR_MULT_TRAIL*atrv
            if (S.pos["trail"] is None) or (trail_stop > S.pos["trail"]):
                S.pos["trail"] = trail_stop
        else:
            trail_stop = price + ATR_MULT_TRAIL*atrv
            if (S.pos["trail"] is None) or (trail_stop < S.pos["trail"]):
                S.pos["trail"] = trail_stop

    # 4) إقفال بالوقوف (trail/breakeven)
    stop = None
    if S.pos["trail"] is not None:
        stop = S.pos["trail"]
    if S.pos["breakeven"] is not None:
        stop = max(stop, S.pos["breakeven"]) if side=="BUY" else min(stop, S.pos["breakeven"]) if stop is not None else S.pos["breakeven"]

    if stop is not None:
        if (side=="BUY" and price <= stop) or (side=="SELL" and price >= stop):
            close_all("Stop (BE/Trail)")
            log(f"{I['trail']} Stop hit @ {price:.6f}", "magenta")

# ---------------------- TV ENTRY ----------------------
def register_tv(signal_dict):
    S.last_tv = signal_dict
    S.last_tv_at = time.time()
    side = signal_dict.get("side")
    log(f"TV signal {side} received", "blue")

# ---------------------- LOGS ----------------------
def log(msg, color=None):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    prefix = colored(now, "white")
    if color:
        print(prefix, colored(msg, color))
    else:
        print(prefix, msg, flush=True)

def log_indicators(df):
    rsi_v = float(rsi(df["close"], RSI_LEN).iloc[-1])
    pdi, mdi, dxv, adxv = adx_dx(df, ADX_LEN)
    dx_last = float(dxv.iloc[-1]); adx_last = float(adxv.iloc[-1])
    pdi_last = float(pdi.iloc[-1]); mdi_last = float(mdi.iloc[-1])
    atr_last = float(atr(df, ATR_LEN).iloc[-1])
    tags = candle_tags(df)
    regime, adx_reg = trend_regime(df)

    buy=False; sell=False  # القرار النهائي لإظهار الحالة فقط (المدخل TV)
    price = float(df["close"].iloc[-1])

    print(f"{I['rsi']} RSI({RSI_LEN}): {rsi_v:.2f}   +DI:{pdi_last:.2f}  -DI:{mdi_last:.2f}  "
          f"{I['dx']} DX:{dx_last:.2f}  {I['adx']} ADX({ADX_LEN}):{adx_last:.2f}  "
          f"{I['atr']} ATR({ATR_LEN}): {atr_last:.6f}")
    print(f"{I['c']} Candles={','.join(tags)}   Regime={regime}   BUY={ '✅' if buy else '❌' }  SELL={ '✅' if sell else '❌' }")
    return rsi_v, adx_last, pdi_last, mdi_last, dx_last, atr_last, tags, regime, price

# ---------------------- ENGINE LOOP ----------------------
def utc_ts(): return datetime.now(timezone.utc).isoformat()

def engine():
    ensure_leverage()
    last_bar_ts = None

    while True:
        try:
            ohlcv = fetch_ohlcv(200)
            df = to_df(ohlcv)
            rsi_v, adx_last, pdi_last, mdi_last, dx_last, atr_last, tags, regime, price = log_indicators(df)

            # --- Keepalive ---
            if time.time() - S.last_keepalive > KEEPALIVE_SECONDS:
                S.last_keepalive = time.time()
                print(colored("keepalive ok (200)", "cyan"))

            # --- Candle timing ---
            bar_ts = df["ts"].iloc[-1]
            new_bar = (last_bar_ts is None) or (bar_ts != last_bar_ts)
            last_bar_ts = bar_ts

            # --- Entry ONLY from TradingView ---
            if FORCE_TV_ENTRIES and S.pos is None:
                if S.last_tv is not None and (time.time()-S.last_tv_at) < 300:
                    side = S.last_tv.get("side")
                    if USE_TV_BAR and not new_bar:
                        # ننتظر إغلاق الشمعة
                        print(colored(f"{I['wait']} waiting bar close for TV entry …", "yellow"))
                    else:
                        open_position(side, price, df)
                        S.last_tv = None  # استهلكنا الإشارة

            # --- After-entry intelligence ---
            manage_after_entry(df)

            # --- Position HUD ---
            if S.pos:
                pside = S.pos["side"]
                rr = (price - S.pos["entry"])/S.pos["entry"] if pside=="BUY" else (S.pos["entry"]-price)/S.pos["entry"]
                rr_pct = rr*100
                trail = S.pos["trail"]
                be = S.pos["breakeven"]
                print(f"{I['pos']} {pside}  entry={S.pos['entry']:.6f}  qty={S.pos['qty']}  PnL%={rr_pct:.2f} "
                      f"BE={'—' if be is None else f'{be:.6f}'}  TRL={'—' if trail is None else f'{trail:.6f}'}")

            else:
                print(f"{I['flat']} FLAT   reason: waiting tv • close in ~?s")

        except Exception as e:
            traceback.print_exc()
            log("engine error: "+str(e), "red")

        time.sleep(DECISION_EVERY_S)

# ---------------------- FLASK ----------------------
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify(ok=True, service="bot", mode=S.mode, time=utc_ts())

@app.get("/metrics")
def metrics():
    p = S.pos or {}
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": S.mode,
        "pos": p,
        "compound_pnl": S.compound_pnl,
        "last_tv": S.last_tv,
        "time": utc_ts()
    })

@app.post("/tv")
def tv():
    """
    توقع الـ JSON من TradingView:
    {
      "pass":"<optional>",
      "side":"BUY" | "SELL",
      "price": 0.2568,
      "timestamp": 169xxx
    }
    """
    try:
        data = request.get_json(force=True)
        side = str(data.get("side","")).upper()
        if side not in ("BUY","SELL"):
            return jsonify(ok=False, err="invalid side"), 400
        price = float(data.get("price", 0)) or 0.0
        register_tv({"side":side, "price":price, "ts": data.get("timestamp", time.time())})
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 400

def run_http():
    app.run(host="0.0.0.0", port=PORT, debug=False)

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    log(f"{I['server']} Serving Flask on :{PORT}  •  Mode={S.mode}", "blue")
    threading.Thread(target=engine, daemon=True).start()
    run_http()
