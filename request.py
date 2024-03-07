import requests

url = 'http://localhost:5000/predict'
data = {
    'n': 2,
    'p': 9,
    'k': 6,
    'humidity': 0,
    'temperature': 0,
    'ph': 0,
    'rainfall': 0
}
r = requests.post(url, json=data)

print(r.json())
