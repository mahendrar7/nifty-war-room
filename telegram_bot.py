"""
telegram_bot.py — Telegram command bot for the Options War Room.

Responds to:
  /status NIFTY   — current NIFTY system snapshot
  /status SENSEX  — current SENSEX system snapshot
  /status         — both instruments
  /enter NIFTY    — log trade entry from last suggestion (or manual)
  /enter NIFTY 23500 CE 150 2  — manual entry (strike type price lots)
  /exit NIFTY     — exit active trade
  /exit SENSEX    — exit active SENSEX trade

Run as a separate process:
  python telegram_bot.py
"""

import json
import os
import time
from datetime import datetime

import requests

from notifier import BOT_TOKEN, CHAT_ID

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
SNAPSHOT_DIR = "data"
CMD_DIR = "data"
POLL_INTERVAL = 2  # seconds


# =============================================================================
# SNAPSHOT READER
# =============================================================================
def read_snapshot(instrument):
    """Read the latest snapshot JSON for an instrument."""
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{instrument.lower()}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def format_snapshot(snap):
    """Format a snapshot dict into a Telegram-friendly message."""
    if snap is None:
        return None

    ts = snap.get("timestamp", "?")
    inst = snap.get("instrument", "?")
    spot = snap.get("spot", 0)
    dte = snap.get("days_to_expiry", "?")

    # Check staleness (> 3 minutes old)
    stale_tag = ""
    try:
        snap_time = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - snap_time).total_seconds()
        if age > 180:
            mins = int(age // 60)
            stale_tag = f"  ⚠️ {mins}m old"
    except ValueError:
        pass

    lines = []
    lines.append(f"*{inst} STATUS*{stale_tag}")
    lines.append(f"_{ts}_")
    lines.append("")

    # Market
    flip = snap.get("flip_level")
    flip_str = ""
    if flip:
        dist = abs(spot - flip)
        flip_str = f"  Flip:{flip} ({dist:.0f}pts)"
    lines.append(f"*Spot:* `{spot}`  ATM: `{snap.get('atm')}`{flip_str}")
    lines.append(f"*MAP:* `{snap.get('put_wall')}` → `{snap.get('gravity')}` → `{snap.get('call_wall')}`")

    straddle = snap.get("straddle", 0)
    em = snap.get("expected_move", 0)
    m5 = snap.get("momentum_5m")
    m5_str = f"  5m: {m5:+.1f}%" if m5 is not None else ""
    lines.append(f"*Straddle:* `{straddle}`  ±`{em}`{m5_str}")
    lines.append(f"*DTE:* {dte}  PCR: {snap.get('pcr', '?')}")
    lines.append("")

    # Regime
    bias = snap.get("bias", "?")
    conf = snap.get("confidence", 0)
    regime = snap.get("regime", "?")
    lines.append(f"*Bias:* {_bias_emoji(bias)} {bias} ({conf}%)")
    lines.append(f"*Regime:* {regime}")
    lines.append(f"*Action:* {snap.get('action', '?')}")

    # Velocity
    vel = snap.get("velocity")
    if vel:
        lines.append(f"*OI Velocity:* {vel}")

    lines.append("")

    # Triggers
    triggers = []
    if "trap" in snap:
        t = snap["trap"]
        triggers.append(f"🪤 {t['type']} {t['confidence']}% → Fade {t['fade_strike']} {t['fade_type']}")
    if "move_prob" in snap:
        mp = snap["move_prob"]
        triggers.append(f"📊 Move Prob: {mp['probability']}% {mp['direction']} ({mp['conviction']})")
    if "vacuum" in snap:
        v = snap["vacuum"]
        triggers.append(f"🌪 Vacuum {v['status']} {v['direction']} Score:{v['score']}")
    if "wall_break" in snap:
        triggers.append(f"🌪 Wall Break {snap['wall_break']['direction']}")
    if "flip_breakout" in snap:
        triggers.append(f"⚡ Flip Breakout {snap['flip_breakout']['direction']}")
    if "liq_accel" in snap:
        la = snap["liq_accel"]
        triggers.append(f"🚀 Accel {la['direction']} {la['conviction']} Score:{la['score']}")

    if triggers:
        lines.append("*— Triggers —*")
        lines.extend(triggers)
        lines.append("")

    # Sniper
    if "sniper" in snap:
        s = snap["sniper"]
        action = s.get("action", "?")
        score = s.get("score", "?")
        direction = s.get("direction", "")
        setup = s.get("setup", "")
        conf = s.get("confidence", "")
        icon = "🎯" if action in ("TAKE TRADE", "SEND IT") else "🔍"
        lines.append(f"{icon} *Sniper:* {action}  Score:{score}/10")
        if setup:
            lines.append(f"   {setup} {direction} ({conf})")
        lines.append("")

    # Active trade + HTL
    if "active_trade" in snap:
        at = snap["active_trade"]
        lines.append(f"*— Active Trade —*")
        lines.append(f"{at['strike']} {at['option_type']} @ ₹{at['entry_price']:.0f} ×{at['lots']}lot  entered {at.get('entry_time', '?')}")
        if "htl" in snap:
            h = snap["htl"]
            v = h["verdict"]
            emoji = {"HOLD": "📈", "TRAIL": "⚠️", "EXIT": "🚨"}.get(v, "")
            lines.append(f"{emoji} HTL: {v}  Score:{h['score']}")
        lines.append("")

    # ML
    if "ml" in snap:
        ml = snap["ml"]
        lines.append(f"🤖 ML: {ml['label']} ({ml['confidence']:.0%}) ±{ml['x_points']:.0f}pts")

    # Heavyweights
    if "heavyweights" in snap:
        hw = snap["heavyweights"]
        stall = " STALLED" if hw.get("stalled") else ""
        trend = f" {hw['roc_trend']}" if hw.get("roc_trend") else ""
        lines.append(f"🏋️ HW: {hw['direction']} {hw['strength']} "
                     f"wROC:{hw['weighted_roc']:+.3f}% "
                     f"aligned:{hw['aligned']}/10{stall}{trend}")

    return "\n".join(lines)


def _bias_emoji(bias):
    if "BULL" in bias:
        return "🟢"
    if "BEAR" in bias:
        return "🔴"
    return "⚪"


# =============================================================================
# TELEGRAM POLLING
# =============================================================================
def get_updates(offset=None):
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(f"{API_BASE}/getUpdates", params=params, timeout=35)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"Poll error: {e}")
    return []


def send_message(chat_id, text):
    try:
        requests.post(f"{API_BASE}/sendMessage", data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })
    except Exception as e:
        print(f"Send error: {e}")


def write_command(instrument, cmd):
    """Write a command JSON for the war room process to pick up."""
    path = os.path.join(CMD_DIR, f"cmd_{instrument.lower()}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, path)


def handle_message(msg):
    chat_id = msg.get("chat", {}).get("id")
    text = (msg.get("text") or "").strip()

    if not text.startswith("/"):
        return

    parts = text.split()
    command = parts[0].lower()

    if command == "/status":
        _handle_status(chat_id, parts)
    elif command == "/enter":
        _handle_enter(chat_id, parts)
    elif command == "/exit":
        _handle_exit(chat_id, parts)
    elif command == "/help":
        send_message(chat_id,
            "*Commands:*\n"
            "/status — system snapshot\n"
            "/status NIFTY — NIFTY only\n"
            "/enter NIFTY — enter from last suggestion\n"
            "/enter NIFTY 23500 CE 150 2 — manual entry\n"
            "/exit NIFTY — exit active trade"
        )


def _handle_status(chat_id, parts):
    instruments = []
    if len(parts) >= 2:
        arg = parts[1].upper()
        if arg in ("NIFTY", "SENSEX"):
            instruments = [arg]
        else:
            send_message(chat_id, f"Unknown instrument: {arg}\nUse: /status NIFTY or /status SENSEX")
            return
    else:
        instruments = ["NIFTY", "SENSEX"]

    for inst in instruments:
        snap = read_snapshot(inst)
        if snap is None:
            send_message(chat_id, f"⚠️ No {inst} data — war room may not be running.")
            continue
        formatted = format_snapshot(snap)
        if formatted:
            send_message(chat_id, formatted)


def _handle_enter(chat_id, parts):
    # /enter NIFTY  or  /enter NIFTY 23500 CE 150 2
    if len(parts) < 2:
        send_message(chat_id, "Usage: /enter NIFTY  or  /enter NIFTY 23500 CE 150 2")
        return

    inst = parts[1].upper()
    if inst not in ("NIFTY", "SENSEX"):
        send_message(chat_id, f"Unknown instrument: {inst}")
        return

    snap = read_snapshot(inst)
    if snap is None:
        send_message(chat_id, f"⚠️ No {inst} data — war room not running.")
        return

    if snap.get("active_trade"):
        at = snap["active_trade"]
        send_message(chat_id,
            f"⚠️ Already in trade: {at['strike']} {at['option_type']} "
            f"@ ₹{at['entry_price']:.0f}")
        return

    if len(parts) >= 6:
        # Manual: /enter NIFTY 23500 CE 150 2
        try:
            strike = int(parts[2])
            opt_type = parts[3].upper()
            price = float(parts[4])
            lots = int(parts[5])
            if opt_type not in ("CE", "PE"):
                send_message(chat_id, "Option type must be CE or PE")
                return
            cmd = {
                "action": "enter",
                "manual": True,
                "strike": strike,
                "option_type": opt_type,
                "price": price,
                "lots": lots,
                "time": datetime.now().strftime("%H:%M"),
            }
            write_command(inst, cmd)
            send_message(chat_id,
                f"✅ Entry queued: {strike} {opt_type} ₹{price:.0f} ×{lots}\n"
                f"Will activate on next {inst} tick.")
        except (ValueError, IndexError):
            send_message(chat_id, "Format: /enter NIFTY 23500 CE 150 2\n(strike type price lots)")
    else:
        # From last suggestion
        cmd = {
            "action": "enter",
            "manual": False,
            "time": datetime.now().strftime("%H:%M"),
        }
        write_command(inst, cmd)
        send_message(chat_id,
            f"✅ Entry from suggestion queued.\n"
            f"Will activate on next {inst} tick.")


def _handle_exit(chat_id, parts):
    if len(parts) < 2:
        send_message(chat_id, "Usage: /exit NIFTY")
        return

    inst = parts[1].upper()
    if inst not in ("NIFTY", "SENSEX"):
        send_message(chat_id, f"Unknown instrument: {inst}")
        return

    snap = read_snapshot(inst)
    if snap is None:
        send_message(chat_id, f"⚠️ No {inst} data — war room not running.")
        return

    if not snap.get("active_trade"):
        send_message(chat_id, f"⚠️ No active {inst} trade to exit.")
        return

    cmd = {
        "action": "exit",
        "time": datetime.now().strftime("%H:%M"),
    }
    write_command(inst, cmd)
    at = snap["active_trade"]
    send_message(chat_id,
        f"🔴 Exit queued: {at['strike']} {at['option_type']} "
        f"@ ₹{at['entry_price']:.0f}\n"
        f"Will exit on next {inst} tick.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🤖 Telegram bot started — /status /enter /exit /help")
    offset = None

    while True:
        updates = get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            if "message" in update:
                handle_message(update["message"])
        if not updates:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
