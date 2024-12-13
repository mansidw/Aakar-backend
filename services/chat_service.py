# routes/chats.py is a blueprint that defines the routes for the chat service.
import uuid
from datetime import datetime
from services.firebase_service import get_firestore_client
db = get_firestore_client()


def get_timestamp():
    return datetime.utcnow().isoformat()

def create_chat_session_if_not_exists(user_id, project_id, query, session_id=None):
    print(f"Creating chat session for user: {user_id}, project: {project_id}, query: {query}")
    if session_id:
        # Check if session exists
        session_ref = db.collection("sessions").document(session_id)
        if session_ref.get().exists:
            return session_id
    # Create new session
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "title": f"Report for {query}",
        "user_id": user_id,
        "project_id": project_id,
        "created_at": get_timestamp(),
    }
    print(f"Creating new chat session: {session_data}")
    saved = db.collection("sessions").document(session_id).set(session_data)
    print(f"Saved: {saved}")
    return session_id

def save_chat(session_id, query, savedFile, report_path, user_id, project_id, report_format, markdown_content):
    # Save chat data to Firebase
    # save file data in base64 format
    file = open(report_path, "rb")
    fileData = file.read()
    file.close()
    chat_data = {
        "session_id": session_id,
        "user_id": user_id,
        "project_id": project_id,
        "query": query,
        "file": fileData,
        "llama_file_id": savedFile.id,
        "format": report_format,
        "timestamp": get_timestamp(),
        "markdown_content": markdown_content
    }
    db.collection("chats").add(chat_data)

def list_sessions(user_id, project_id):
    sessions_ref = db.collection("sessions").where("user_id", "==", user_id).where("project_id", "==", project_id)
    sessions = [doc.to_dict() for doc in sessions_ref.stream()]
    return sessions

import base64

def list_chats(session_id):
    chats_ref = db.collection("chats").where("session_id", "==", session_id).order_by("timestamp")
    chats = []
    for doc in chats_ref.stream():
        chat = doc.to_dict()
        if "file" in chat and chat["file"]:  # Check if the file field exists
            # Encode the file field to Base64
            chat["file"] = base64.b64encode(chat["file"]).decode("utf-8")
        chats.append(chat)
    return chats


def upload_files(projectid, fileid):
    file = {
        "project_id": projectid,
        "file_id": fileid,
        "timestamp": get_timestamp()
    }
    db.collection("files").add(file)
    return file

def get_files_project(projectid):
    files_ref = db.collection("files").where("project_id", "==", projectid)
    files = [doc.to_dict() for doc in files_ref.stream()]
    return files

def delete_session_chats(session_id):
    # Delete session
    try:
        db.collection("sessions").document(session_id).delete()

        # Delete chats
        chats_ref = db.collection("chats").where("session_id", "==", session_id)
        return True
    except Exception as e:
        print(f"Error deleting session: {e}")
        return False