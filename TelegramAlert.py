import requests
import time
# Your Telegram Bot API Token
token = ''

# Your personal chat ID
chat_id = ''
message = 'Alert! Drowning detected near the premises'
# Method to send a message to your Telegram account
def send_telegram_message():
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, json=payload)
    return response.json()

# # Call the function with your alert message
# for _ in range(5):
#     alert_message = 'Alert! Drowning detected near the premises'
#     response = send_telegram_message(alert_message)
#     print(response)


# # Method to get your chat ID
# def get_chat_id():
#     url = f'https://api.telegram.org/bot{token}/getUpdates'
#     response = requests.get(url)
#     data = response.json()
#     if 'result' in data and data['result']:
#         chat_id = data['result'][0]['message']['chat']['id']
#         return chat_id
#     else:
#         return None

# # Call the function to get your chat ID
# chat_id = get_chat_id()
# print("Your chat ID:", chat_id)
send_telegram_message()