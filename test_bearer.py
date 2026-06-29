import requests
import json

url = "https://www.kaggle.com/api/v1/competitions/data/download-all/neural-debris-removal-in-streak-detection-models"
token = "KGAT_aadac020c5ae44cde5ad90a898e009c7"
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(url, headers=headers, stream=True)
print(response.status_code)
if response.status_code != 200:
    print(response.text)
else:
    print("Success!")
