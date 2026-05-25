import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RUNNER_BOT_TOKEN = os.getenv("TELEGRAM_RUNNER_BOT_TOKEN", "")
RUNNER_CHAT_ID = os.getenv("TELEGRAM_RUNNER_CHAT_ID", "")

def _post(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": chat_id, "text": message})
        if response.status_code != 200:
            print("❌ Telegram Error:", response.text)
    except Exception as e:
        print("⚠️ Telegram Exception:", str(e))

def send_telegram_message(message):
    _post(BOT_TOKEN, CHAT_ID, message)

def send_runner_alert(message):
    _post(RUNNER_BOT_TOKEN, RUNNER_CHAT_ID, message)
