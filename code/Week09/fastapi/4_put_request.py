import requests

url = "YOUR_BASE_URL"  # https://abc-def.ngrok.io

update_item = {"id": 3, "name": "9 UP", "price": 90}

response = requests.put(f"{url}/items/{update_item['id']}", json=update_item)

print("Put Result:")
print(response.json())

response = requests.get(f"{url}/items/{update_item['id']}")

print("Get Result:")
print(response.json())
