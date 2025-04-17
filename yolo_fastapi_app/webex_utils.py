# webex_utils.py
import requests

WEBEX_WEBHOOK_URL = "https://webexapis.com/v1/webhooks/incoming/Y2lzY29zcGFyazovL3VybjpURUFNOnVzLXdlc3QtMl9yL1dFQkhPT0svMDQyYTE1NTItNWFiYS00OTkzLTliYzItM2IyZGNjOTNmNzMx"

def send_webex_message(message: str):
    data = {"text": message}
    headers = {"Content-Type": "application/json"}
    response = requests.post(WEBEX_WEBHOOK_URL, json=data, headers=headers)
