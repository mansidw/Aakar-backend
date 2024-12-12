# routes/chats.py is a blueprint that defines the routes for the chat service.
import uuid
from datetime import datetime
from services.firebase_service import get_firestore_client
db = get_firestore_client()

def get_timestamp():
    return datetime.utcnow().isoformat()

def create_chat_session_if_not_exists(user_id, project_id, session_id=None):
    if session_id:
        # Check if session exists
        session_ref = db.collection("sessions").document(session_id)
        if session_ref.get().exists:
            return session_id
    # Create new session
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "title": "New Session",
        "user_id": user_id,
        "project_id": project_id,
        "created_at": get_timestamp(),
    }
    db.collection("sessions").document(session_id).set(session_data)
    return session_id

def save_chat(session_id, query, response):
    chat_data = {
        "session_id": session_id,
        "query": query,
        "response": response,
        "timestamp": get_timestamp(),
    }
    db.collection("chats").add(chat_data)

def list_sessions(user_id):
    sessions_ref = db.collection("sessions").where("user_id", "==", user_id)
    sessions = [doc.to_dict() for doc in sessions_ref.stream()]
    return sessions

def list_chats(session_id):
    chats_ref = db.collection("chats").where("session_id", "==", session_id).order_by("timestamp")
    chats = [doc.to_dict() for doc in chats_ref.stream()]
    return chats
