#!/bin/bash
# End-of-day ML retrain — runs both NIFTY and SENSEX
# Scheduled via cron at 15:45 Mon–Fri

cd /Users/mahendra/PycharmProjects/nifty_war_room

LOG="data/retrain_eod.log"
echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG"

echo "Retraining NIFTY..." >> "$LOG"
.venv/bin/python ml_engine.py retrain nifty >> "$LOG" 2>&1
echo "Retraining SENSEX..." >> "$LOG"
.venv/bin/python ml_engine.py retrain sensex >> "$LOG" 2>&1

echo "Done." >> "$LOG"
