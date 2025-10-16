import requests

url = "YOUR_BASE_URL"  # https://abc-def.ngrok.io

new_item = {"id": 4, "name": "Orange Juice", "price": 10}

response = requests.post(f"{url}/items", json=new_item)

print("Post Result:")
print(response.json())

response = requests.get(f"{url}/items/{new_item['id']}")

print("Get Result:")
print(response.json())
