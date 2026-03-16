import os
import threading
import webbrowser
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

# Configuration
API_KEY = "vgq6rzgyww3lzbiz"
API_SECRET = "zx0v9nqvpwzlf1b34vwjpi1o5201w62p"
ACCESS_TOKEN_PATH = "access_token.txt"
PORT = 8080  # Port 80 often requires admin; 8080 is safer for local dev

# Shared data between threads
auth_data = {"request_token": None}


class TokenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        if "request_token" in query_components:
            # Update the shared variable
            auth_data["request_token"] = query_components["request_token"][0]

            # Send response to browser
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Success!</h1><p>Token captured. You can close this window.</p>")

            # Shut down the server immediately
            threading.Thread(target=self.server.shutdown).start()


def start_background_listener():
    """Initializes and runs the server in a separate thread."""
    server = HTTPServer(('127.0.0.1', PORT), TokenHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def get_kite_client():
    kite = KiteConnect(api_key=API_KEY)

    # 1. Load existing token if valid
    if os.path.exists(ACCESS_TOKEN_PATH):
        with open(ACCESS_TOKEN_PATH, "r") as f:
            access_token = f.read().strip()
        kite.set_access_token(access_token)
        try:
            kite.profile()
            print("🔑 Access token valid.")
            return kite
        except:
            print("⚠️ Token expired.")

    # 2. Start Background Listener
    print(f"📡 Starting background listener on port {PORT}...")
    start_background_listener()

    # 3. Open Login URL
    login_url = kite.login_url()
    webbrowser.open(login_url)
    print(f"🌍 Please log in here: {login_url}")

    # 4. Do other work while waiting
    print("⏳ Waiting for login... (You can run other non-blocking tasks here)")
    while auth_data["request_token"] is None:
        # Mocking "other work" the main thread could do
        time.sleep(1)

        # 5. Process the captured token
    token = auth_data["request_token"]
    print(f"✅ Token received: {token}")

    session = kite.generate_session(token, api_secret=API_SECRET)
    access_token = session["access_token"]

    with open(ACCESS_TOKEN_PATH, "w") as f:
        f.write(access_token)

    kite.set_access_token(access_token)
    return kite


if __name__ == "__main__":
    client = get_kite_client()
    print("🚀 Kite Client is ready for use!")
