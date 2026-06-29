import os
import requests
import kagglehub
from kagglehub.clients import KaggleApiV1Client

# Monkey patch kagglehub to support the new Kaggle API Tokens (Bearer tokens)
class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r

original_get_auth = KaggleApiV1Client._get_auth
def _patched_get_auth(self):
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return BearerAuth(token)
    return original_get_auth(self)

KaggleApiV1Client._get_auth = _patched_get_auth

# Download latest version
path = kagglehub.competition_download('neural-debris-removal-in-streak-detection-models')

print("Path to competition files:", path)
