import requests

url = "YOUR_BASE_URL"  # https://abc-def.ngrok.io

response = requests.get(f"{url}/")

print("Get Result:")
print(response.json())
