"""
trade_logger.py — Lightweight Flask app for logging trade entries/exits.

Replaces Telegram-based entry/exit with a browser UI.
Writes commands to data/cmd_{instrument}.json (same format as telegram_bot.py)
so the war room picks them up on the next tick.

Also maintains data/trade_log.csv for historical tracking.

Run:
    python trade_logger.py
    → opens http://localhost:5050

Auth:
    Set env vars TRADE_LOGGER_USER and TRADE_LOGGER_PASS to enable basic auth.
    e.g.  TRADE_LOGGER_USER=mahendra TRADE_LOGGER_PASS=secret python trade_logger.py
"""

import csv
import json
import os
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, render_template_string, request, Response

from config import INSTRUMENT_PROFILES

app = Flask(__name__)

# =============================================================================
# BASIC AUTH
# =============================================================================
AUTH_USER = os.environ.get("TRADE_LOGGER_USER", "")
AUTH_PASS = os.environ.get("TRADE_LOGGER_PASS", "")


def check_auth(username, password):
    return username == AUTH_USER and password == AUTH_PASS


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_USER:
            return f(*args, **kwargs)  # no auth configured, skip
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                "Login required.", 401,
                {"WWW-Authenticate": 'Basic realm="Trade Logger"'},
            )
        return f(*args, **kwargs)
    return decorated

DATA_DIR = "data"
TRADE_LOG = os.path.join(DATA_DIR, "trade_log.csv")
TRADE_LOG_COLUMNS = [
    "timestamp", "instrument", "action", "strike", "option_type",
    "price", "lots", "entry_time", "exit_time", "pnl_per_lot", "notes",
]


# =============================================================================
# HELPERS
# =============================================================================

def read_snapshot(instrument):
    path = os.path.join(DATA_DIR, f"snapshot_{instrument.lower()}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def write_command(instrument, cmd):
    path = os.path.join(DATA_DIR, f"cmd_{instrument.lower()}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, path)


def append_trade_log(row):
    file_exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_trade_log(limit=50):
    if not os.path.exists(TRADE_LOG):
        return []
    try:
        with open(TRADE_LOG) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-limit:][::-1]  # most recent first
    except Exception:
        return []


def get_strikes(instrument):
    """Get available strikes from snapshot."""
    snap = read_snapshot(instrument)
    if not snap:
        profile = INSTRUMENT_PROFILES.get(instrument, {})
        step = profile.get("strike_step", 50)
        # fallback: generate around a round number
        base = 23500 if instrument == "NIFTY" else 77000
        return [base + i * step for i in range(-10, 11)]
    spot = snap.get("spot", 0)
    profile = INSTRUMENT_PROFILES.get(instrument, {})
    step = profile.get("strike_step", 50)
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-10, 11)]


# =============================================================================
# API ROUTES
# =============================================================================

@app.route("/api/state/<instrument>")
@auth_required
def api_state(instrument):
    instrument = instrument.upper()
    snap = read_snapshot(instrument)
    strikes = get_strikes(instrument)
    profile = INSTRUMENT_PROFILES.get(instrument, {})

    result = {
        "instrument": instrument,
        "strikes": strikes,
        "spot": snap.get("spot") if snap else None,
        "active_trade": snap.get("active_trade") if snap else None,
        "htl": snap.get("htl") if snap else None,
        "sniper": snap.get("sniper") if snap else None,
        "suggestion": snap.get("suggestion") if snap else None,
        "lot_size": profile.get("lot_size", 65),
        "heavyweights": snap.get("heavyweights") if snap else None,
        "bias": snap.get("bias") if snap else None,
        "action": snap.get("action") if snap else None,
        "call_wall": snap.get("call_wall") if snap else None,
        "put_wall": snap.get("put_wall") if snap else None,
        "gravity": snap.get("gravity") if snap else None,
    }
    return jsonify(result)


@app.route("/api/enter", methods=["POST"])
@auth_required
def api_enter():
    data = request.json
    instrument = data.get("instrument", "").upper()
    if instrument not in INSTRUMENT_PROFILES:
        return jsonify({"error": "Invalid instrument"}), 400

    strike = int(data["strike"])
    option_type = data["option_type"].upper()
    price = float(data["price"])
    lots = int(data["lots"])
    notes = data.get("notes", "")

    if option_type not in ("CE", "PE"):
        return jsonify({"error": "Option type must be CE or PE"}), 400

    now_str = datetime.now().strftime("%H:%M")

    cmd = {
        "action": "enter",
        "manual": True,
        "strike": strike,
        "option_type": option_type,
        "price": price,
        "lots": lots,
        "time": now_str,
    }
    write_command(instrument, cmd)

    append_trade_log({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instrument": instrument,
        "action": "ENTER",
        "strike": strike,
        "option_type": option_type,
        "price": price,
        "lots": lots,
        "entry_time": now_str,
        "exit_time": "",
        "pnl_per_lot": "",
        "notes": notes,
    })

    return jsonify({
        "status": "ok",
        "message": f"Entry queued: {strike} {option_type} ₹{price:.0f} ×{lots}",
    })


@app.route("/api/exit", methods=["POST"])
@auth_required
def api_exit():
    data = request.json
    instrument = data.get("instrument", "").upper()
    if instrument not in INSTRUMENT_PROFILES:
        return jsonify({"error": "Invalid instrument"}), 400

    exit_price = data.get("exit_price")
    notes = data.get("notes", "")

    now_str = datetime.now().strftime("%H:%M")

    cmd = {
        "action": "exit",
        "time": now_str,
    }
    write_command(instrument, cmd)

    # Log exit
    snap = read_snapshot(instrument)
    at = snap.get("active_trade") if snap else None
    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instrument": instrument,
        "action": "EXIT",
        "strike": at["strike"] if at else "",
        "option_type": at["option_type"] if at else "",
        "price": exit_price or "",
        "lots": at["lots"] if at else "",
        "entry_time": at.get("entry_time", "") if at else "",
        "exit_time": now_str,
        "pnl_per_lot": "",
        "notes": notes,
    }

    if exit_price and at and at.get("entry_price"):
        try:
            log_row["pnl_per_lot"] = round(float(exit_price) - float(at["entry_price"]), 2)
        except (ValueError, TypeError):
            pass

    append_trade_log(log_row)

    return jsonify({
        "status": "ok",
        "message": f"Exit queued for {instrument}",
    })


@app.route("/api/history")
@auth_required
def api_history():
    rows = read_trade_log(100)  # read more to pair up
    # Club entry+exit into single trades for display
    # rows are most-recent-first — process in that order so each EXIT
    # gets matched to the ENTER that comes right after it (i.e. the
    # most recent entry before that exit chronologically)
    trades = []
    pending_exits = {}  # instrument -> list of exit rows waiting for a match
    for r in rows:
        if r["action"] == "EXIT":
            inst_key = r.get("instrument", "")
            pending_exits.setdefault(inst_key, []).append(r)
        elif r["action"] == "ENTER":
            inst_key = r.get("instrument", "")
            exit_row = None
            if inst_key in pending_exits and pending_exits[inst_key]:
                exit_row = pending_exits[inst_key].pop(0)

            entry_price = float(r.get("price", 0) or 0)
            exit_price = float(exit_row.get("price", 0) or 0) if exit_row else 0
            pnl = None
            if exit_row and exit_price:
                lot_size = INSTRUMENT_PROFILES.get(inst_key, {}).get("lot_size", 65)
                lots_val = int(r.get("lots", 1) or 1)
                gross = (exit_price - entry_price) * lot_size * lots_val
                turnover = (entry_price + exit_price) * lot_size * lots_val
                brokerage = 40  # flat per round trip
                commission = turnover * 0.037
                pnl = round(gross - brokerage - commission, 2)

            trade = {
                "instrument": inst_key,
                "strike": r.get("strike", ""),
                "option_type": r.get("option_type", ""),
                "entry_price": r.get("price", ""),
                "entry_time": r.get("entry_time") or (r.get("timestamp", "").split(" ")[1][:5] if r.get("timestamp") else ""),
                "lots": r.get("lots", ""),
                "exit_price": exit_row.get("price", "") if exit_row else "",
                "exit_time": exit_row.get("exit_time") or (exit_row.get("timestamp", "").split(" ")[1][:5] if exit_row and exit_row.get("timestamp") else "") if exit_row else "",
                "pnl_per_lot": pnl,
                "status": "CLOSED" if exit_row else "OPEN",
                "notes": r.get("notes", ""),
            }
            trades.append(trade)

    return jsonify(trades[:50])


# =============================================================================
# UI
# =============================================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trade Logger</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace;
    background: #0d1117; color: #c9d1d9;
    padding: 12px; max-width: 480px; margin: 0 auto;
  }
  h1 { font-size: 16px; color: #58a6ff; margin-bottom: 12px; }
  h2 { font-size: 13px; color: #8b949e; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }

  .info-snapshot {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 12px; font-size: 12px;
  }
  .info-row { display: flex; align-items: center; padding: 3px 0; }
  .info-label {
    width: 36px; flex-shrink: 0; color: #8b949e; font-weight: 700;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .info-val { color: #c9d1d9; font-size: 12px; }
  .info-val .bullish { color: #3fb950; }
  .info-val .bearish { color: #f85149; }
  .info-val .neutral { color: #d29922; }

  .tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .tab {
    flex: 1; padding: 8px; text-align: center; border: 1px solid #30363d;
    background: #161b22; color: #8b949e; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 600;
  }
  .tab.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }

  .inst-tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .inst-tab {
    flex: 1; padding: 6px; text-align: center; border: 1px solid #30363d;
    background: #161b22; color: #8b949e; border-radius: 6px; cursor: pointer;
    font-size: 12px; font-weight: 600;
  }
  .inst-tab.active { background: #238636; color: #fff; border-color: #238636; }


  .active-trade {
    background: #1a2332; border: 1px solid #1f6feb; border-radius: 6px;
    padding: 10px 12px; margin-bottom: 12px; font-size: 13px;
  }
  .active-trade .label { color: #3fb950; font-weight: 700; margin-bottom: 4px; }
  .active-trade .details { color: #c9d1d9; }

  .htl-bar {
    display: flex; align-items: center; gap: 10px;
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 12px; margin-bottom: 12px;
  }
  .htl-dot {
    width: 14px; height: 14px; border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 1.5s ease-in-out infinite;
  }
  .htl-dot.hold { background: #3fb950; box-shadow: 0 0 8px #3fb95088; }
  .htl-dot.trail { background: #d29922; box-shadow: 0 0 8px #d2992288; }
  .htl-dot.exit { background: #f85149; box-shadow: 0 0 8px #f8514988; animation: pulse-fast 0.6s ease-in-out infinite; }
  .htl-dot.none { background: #484f58; animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  @keyframes pulse-fast { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .htl-verdict { font-weight: 700; font-size: 14px; }
  .htl-verdict.hold { color: #3fb950; }
  .htl-verdict.trail { color: #d29922; }
  .htl-verdict.exit { color: #f85149; }
  .htl-score { color: #8b949e; font-size: 12px; }
  .htl-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }

  .form-group { margin-bottom: 10px; }
  label { display: block; font-size: 11px; color: #8b949e; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
  select, input {
    width: 100%; padding: 8px 10px; background: #161b22; border: 1px solid #30363d;
    color: #c9d1d9; border-radius: 6px; font-size: 14px; font-family: inherit;
  }
  select:focus, input:focus { border-color: #1f6feb; outline: none; }

  .row { display: flex; gap: 8px; }
  .row > .form-group { flex: 1; }

  .dir-btns { display: flex; gap: 8px; margin-bottom: 10px; }
  .dir-btn {
    flex: 1; padding: 10px; text-align: center; border: 2px solid #30363d;
    background: #161b22; border-radius: 6px; cursor: pointer;
    font-size: 14px; font-weight: 700;
  }
  .dir-btn.ce { color: #3fb950; }
  .dir-btn.pe { color: #f85149; }
  .dir-btn.ce.active { border-color: #3fb950; background: #0d2818; }
  .dir-btn.pe.active { border-color: #f85149; background: #2d1215; }

  .btn-enter {
    width: 100%; padding: 12px; background: #238636; color: #fff;
    border: none; border-radius: 6px; font-size: 15px; font-weight: 700;
    cursor: pointer; margin-top: 8px;
  }
  .btn-enter:hover { background: #2ea043; }
  .btn-enter:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }

  .btn-exit {
    width: 100%; padding: 12px; background: #da3633; color: #fff;
    border: none; border-radius: 6px; font-size: 15px; font-weight: 700;
    cursor: pointer; margin-top: 8px;
  }
  .btn-exit:hover { background: #f85149; }

  .toast {
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    background: #238636; color: #fff; padding: 10px 20px; border-radius: 6px;
    font-size: 13px; font-weight: 600; display: none; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .toast.error { background: #da3633; }

  .history-table { width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 8px; }
  .history-table th { color: #8b949e; text-align: left; padding: 4px 6px; border-bottom: 1px solid #30363d; }
  .history-table td { padding: 4px 6px; border-bottom: 1px solid #21262d; }
  .history-table .enter { color: #3fb950; }
  .history-table .exit { color: #f85149; }
  .pnl-pos { color: #3fb950; }
  .pnl-neg { color: #f85149; }

  .section { display: none; }
  .section.active { display: block; }
</style>
</head>
<body>

<div class="inst-tabs">
  <div class="inst-tab active" data-inst="NIFTY" onclick="switchInst('NIFTY')">NIFTY</div>
  <div class="inst-tab" data-inst="SENSEX" onclick="switchInst('SENSEX')">SENSEX</div>
</div>

<div class="info-snapshot" id="info-snapshot">
  <div class="info-row">
    <span class="info-label">HW</span>
    <span class="info-val" id="info-hw">--</span>
  </div>
  <div class="info-row">
    <span class="info-label">MAP</span>
    <span class="info-val" id="info-map">--</span>
  </div>
  <div class="info-row">
    <span class="info-label">REC</span>
    <span class="info-val" id="info-rec">--</span>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('entry')">Entry</div>
  <div class="tab" onclick="switchTab('exit')">Exit</div>
  <div class="tab" onclick="switchTab('history')">History</div>
</div>

<div id="active-trade-box" class="active-trade" style="display:none">
  <div class="label">ACTIVE TRADE</div>
  <div class="details" id="active-trade-details"></div>
</div>

<div id="htl-bar" class="htl-bar" style="display:none">
  <div class="htl-dot none" id="htl-dot"></div>
  <div>
    <span class="htl-label">HTL </span>
    <span class="htl-verdict" id="htl-verdict">--</span>
    <span class="htl-score" id="htl-score"></span>
  </div>
</div>

<div id="toast" class="toast"></div>

<!-- ENTRY SECTION -->
<div id="section-entry" class="section active">
  <div class="dir-btns">
    <div class="dir-btn ce active" id="btn-ce" onclick="setDir('CE')">CE (CALL)</div>
    <div class="dir-btn pe" id="btn-pe" onclick="setDir('PE')">PE (PUT)</div>
  </div>

  <div class="row">
    <div class="form-group">
      <label>Strike</label>
      <select id="strike"></select>
    </div>
    <div class="form-group">
      <label>Lots</label>
      <select id="lots"></select>
    </div>
  </div>

  <div class="form-group">
    <label>Entry Price (₹)</label>
    <input type="number" id="entry-price" step="0.5" placeholder="option premium">
  </div>

  <div class="form-group">
    <label>Notes (optional)</label>
    <input type="text" id="entry-notes" placeholder="setup, reason...">
  </div>

  <button class="btn-enter" id="btn-submit-entry" onclick="submitEntry()">ENTER TRADE</button>
</div>

<!-- EXIT SECTION -->
<div id="section-exit" class="section">
  <div class="form-group">
    <label>Exit Price (₹, optional)</label>
    <input type="number" id="exit-price" step="0.5" placeholder="for P&L tracking">
  </div>
  <div class="form-group">
    <label>Notes (optional)</label>
    <input type="text" id="exit-notes" placeholder="reason for exit...">
  </div>
  <button class="btn-exit" id="btn-submit-exit" onclick="submitExit()">EXIT TRADE</button>
</div>

<!-- HISTORY SECTION -->
<div id="section-history" class="section">
  <table class="history-table">
    <thead>
      <tr><th>Strike</th><th>Entry</th><th>Exit</th><th>Lots</th><th>Net P&L</th><th>Status</th></tr>
    </thead>
    <tbody id="history-body"></tbody>
  </table>
</div>

<script>
let currentInst = 'NIFTY';
let currentDir = 'CE';
let stateData = {};

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', t.textContent.toLowerCase() === tab);
  });
  ['entry', 'exit', 'history'].forEach(s => {
    document.getElementById('section-' + s).classList.toggle('active', s === tab);
  });
  if (tab === 'history') loadHistory();
}

function switchInst(inst) {
  currentInst = inst;
  document.querySelectorAll('.inst-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.inst === inst);
  });
  loadState();
}

function setDir(dir) {
  currentDir = dir;
  document.getElementById('btn-ce').classList.toggle('active', dir === 'CE');
  document.getElementById('btn-pe').classList.toggle('active', dir === 'PE');
}

function toast(msg, isError) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast' + (isError ? ' error' : '');
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}

async function loadState() {
  try {
    const res = await fetch('/api/state/' + currentInst);
    stateData = await res.json();

    // Info snapshot — 3 lines
    const hw = stateData.heavyweights;
    if (hw) {
      const dirCls = hw.direction === 'BULLISH' ? 'bullish' : hw.direction === 'BEARISH' ? 'bearish' : 'neutral';
      const movers = (hw.top_movers || []).map(m => {
        const sign = m.roc >= 0 ? '+' : '';
        return `${m.name} ${sign}${m.roc.toFixed(1)}%`;
      }).join(', ');
      document.getElementById('info-hw').innerHTML =
        `<span class="${dirCls}">${hw.direction}</span> ${hw.strength} | ${movers}`;
    } else {
      document.getElementById('info-hw').textContent = '--';
    }

    // MAP line: call wall, put wall, gravity
    if (stateData.spot) {
      const cw = stateData.call_wall || '--';
      const pw = stateData.put_wall || '--';
      const grav = stateData.gravity || '--';
      document.getElementById('info-map').innerHTML =
        `Spot <strong>${stateData.spot.toFixed(0)}</strong> | CW ${cw} · PW ${pw} · Grav ${grav}`;
    } else {
      document.getElementById('info-map').textContent = '--';
    }

    // REC line: sniper action + bias
    const sniper = stateData.sniper;
    if (sniper) {
      const dirCls = sniper.direction === 'BULLISH' ? 'bullish' : sniper.direction === 'BEARISH' ? 'bearish' : 'neutral';
      document.getElementById('info-rec').innerHTML =
        `<span class="${dirCls}">${sniper.action}</span> | ${stateData.bias || '--'} · ${stateData.action || ''}`;
    } else {
      document.getElementById('info-rec').textContent = stateData.bias || '--';
    }

    // Populate strikes
    const sel = document.getElementById('strike');
    const oldVal = sel.value;
    sel.innerHTML = '';
    const strikes = stateData.strikes || [];
    const spot = stateData.spot || 0;
    let closest = strikes[0];
    strikes.forEach(s => {
      if (Math.abs(s - spot) < Math.abs(closest - spot)) closest = s;
    });
    strikes.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      if (s === closest) opt.selected = true;
      sel.appendChild(opt);
    });
    if (oldVal && strikes.includes(parseInt(oldVal))) sel.value = oldVal;

    // Populate lots dropdown (once)
    const lotsSel = document.getElementById('lots');
    if (lotsSel.options.length === 0) {
      for (let i = 1; i <= 10; i++) {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = i;
        if (i === 2) opt.selected = true;
        lotsSel.appendChild(opt);
      }
    }

    // Active trade
    const atBox = document.getElementById('active-trade-box');
    const at = stateData.active_trade;
    if (at) {
      atBox.style.display = 'block';
      document.getElementById('active-trade-details').textContent =
        `${at.strike} ${at.option_type} @ ₹${at.entry_price} × ${at.lots} lot | ${at.entry_time || ''}`;
    } else {
      atBox.style.display = 'none';
    }

    // HTL status
    const htlBar = document.getElementById('htl-bar');
    const htl = stateData.htl;
    if (at && htl) {
      htlBar.style.display = 'flex';
      const verdict = (htl.verdict || '').toUpperCase();
      const cls = verdict === 'HOLD' ? 'hold' : verdict === 'TRAIL' ? 'trail' : verdict === 'EXIT' ? 'exit' : 'none';
      document.getElementById('htl-dot').className = 'htl-dot ' + cls;
      const verdictEl = document.getElementById('htl-verdict');
      verdictEl.textContent = verdict || '--';
      verdictEl.className = 'htl-verdict ' + cls;
      document.getElementById('htl-score').textContent =
        htl.score != null ? `  score: ${htl.score}` : '';
    } else if (at) {
      htlBar.style.display = 'flex';
      document.getElementById('htl-dot').className = 'htl-dot hold';
      const verdictEl = document.getElementById('htl-verdict');
      verdictEl.textContent = 'ACTIVE';
      verdictEl.className = 'htl-verdict hold';
      document.getElementById('htl-score').textContent = '';
    } else {
      htlBar.style.display = 'none';
    }

    // Disable entry if already in trade
    document.getElementById('btn-submit-entry').disabled = !!at;

  } catch (e) {
    document.getElementById('info-hw').textContent = 'offline';
    document.getElementById('info-map').textContent = '--';
    document.getElementById('info-rec').textContent = '--';
  }
}

async function submitEntry() {
  const strike = document.getElementById('strike').value;
  const price = document.getElementById('entry-price').value;
  const lots = document.getElementById('lots').value;
  const notes = document.getElementById('entry-notes').value;

  if (!price) { toast('Enter a price', true); return; }

  try {
    const res = await fetch('/api/enter', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        instrument: currentInst,
        strike: strike,
        option_type: currentDir,
        price: price,
        lots: lots,
        notes: notes,
      }),
    });
    const data = await res.json();
    if (data.error) { toast(data.error, true); return; }
    toast(data.message);
    document.getElementById('entry-price').value = '';
    document.getElementById('entry-notes').value = '';
    setTimeout(loadState, 1500);
  } catch (e) {
    toast('Failed: ' + e.message, true);
  }
}

async function submitExit() {
  const exitPrice = document.getElementById('exit-price').value;
  const notes = document.getElementById('exit-notes').value;

  try {
    const res = await fetch('/api/exit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        instrument: currentInst,
        exit_price: exitPrice || null,
        notes: notes,
      }),
    });
    const data = await res.json();
    if (data.error) { toast(data.error, true); return; }
    toast(data.message);
    document.getElementById('exit-price').value = '';
    document.getElementById('exit-notes').value = '';
    setTimeout(loadState, 1500);
  } catch (e) {
    toast('Failed: ' + e.message, true);
  }
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const trades = await res.json();
    const tbody = document.getElementById('history-body');
    tbody.innerHTML = '';
    trades.forEach(t => {
      const tr = document.createElement('tr');
      let pnlHtml = '';
      if (t.pnl_per_lot != null) {
        const cls = t.pnl_per_lot >= 0 ? 'pnl-pos' : 'pnl-neg';
        pnlHtml = `<span class="${cls}">${t.pnl_per_lot >= 0 ? '+' : ''}₹${t.pnl_per_lot}</span>`;
      }
      const statusCls = t.status === 'OPEN' ? 'enter' : '';
      tr.innerHTML = `
        <td>${t.strike} ${t.option_type}</td>
        <td>${t.entry_price} @ ${t.entry_time}</td>
        <td>${t.exit_price ? t.exit_price + ' @ ' + t.exit_time : '--'}</td>
        <td>${t.lots}</td>
        <td>${pnlHtml}</td>
        <td class="${statusCls}">${t.status}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {}
}

// Initial load + auto-refresh (3s for live HTL tracking)
loadState();
setInterval(loadState, 3000);
</script>
</body>
</html>
"""


@app.route("/")
@auth_required
def index():
    return render_template_string(HTML)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Trade Logger → http://localhost:5050")
    if AUTH_USER:
        print(f"🔒 Auth enabled (user: {AUTH_USER})")
    else:
        print("⚠  No auth — set TRADE_LOGGER_USER & TRADE_LOGGER_PASS for ngrok use")
    app.run(host="0.0.0.0", port=5050, debug=False)
