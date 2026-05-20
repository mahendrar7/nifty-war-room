import requests

BOT_TOKEN = "7702090637:AAHTv7qUNXTLaKZ4eO31Hwn_jZ9xkQY--vQ"
CHAT_ID = "1476855939"

RUNNER_BOT_TOKEN = "8841093148:AAE9g4LPSJcP6ckOAw6356Er84Si6f4sXfo"
RUNNER_CHAT_ID = "-1003886854270"

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
