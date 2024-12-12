#config.py
import os
from firebase_admin import credentials, initialize_app

# Firebase Configuration
FIREBASE_CREDENTIALS = "./firebase_credentials.json"  # Path to your Firebase private key
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_app = initialize_app(cred)

# Llama Cloud Configuration
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
