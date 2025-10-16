import requests

url = "YOUR_BASE_URL"  # https://abc-def.ngrok.io

item_id = 1

response = requests.get(f"{url}/items/{item_id}")

print("Get Result:")
print(response.json())
