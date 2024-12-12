# services/firebase_service.py - Firebase service to initialize Firestore client
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore_client():
    """
    Initialize Firebase Firestore client.
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate("./firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()
