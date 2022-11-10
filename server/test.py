import requests

resp = requests.post(url="http://127.0.0.1:5000/predict", json=[
    [2107571221, 675, 0, 28993524843]
])

print(resp.text)
