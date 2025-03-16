import requests

TOKEN = "7893108698:AAF2gJWWWYsyU7SNfKRIcjfviBrlo-mILto"  # Ganti dengan Token yang didapat dari BotFather
CHAT_ID = "722696156"  # Ganti dengan chat ID yang didapat dari langkah sebelumnya

MESSAGE = "🚨 Alert! Pengunjung RPTRA sudah mencapai 50% kapasitas"

URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
PARAMS = {"chat_id": CHAT_ID, "text": MESSAGE}

response = requests.get(URL, params=PARAMS)
print(response.json())  # Cek respons dari Telegram
