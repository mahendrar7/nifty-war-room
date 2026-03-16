import requests

BOT_TOKEN = "7702090637:AAHTv7qUNXTLaKZ4eO31Hwn_jZ9xkQY--vQ"
CHAT_ID = "1476855939"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print("❌ Telegram Error:", response.text)
    except Exception as e:
        print("⚠️ Telegram Exception:", str(e))
