# routes/chats.py is a blueprint that defines the routes for the chat service.
from flask import Blueprint, jsonify, request
from services.chat_service import list_sessions, list_chats, delete_session_chats

chats_bp = Blueprint("chats", __name__)

@chats_bp.route("/sessions", methods=["GET"])
def get_sessions():
    user_id = request.args.get("user_id")
    project_id = request.args.get("project_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    sessions = list_sessions(user_id, project_id)
    return jsonify(sessions), 200

@chats_bp.route("/session/<session_id>", methods=["GET"])
def get_chats_in_session(session_id):
    chats = list_chats(session_id)
    return jsonify(chats), 200

# delete session
@chats_bp.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    delete_session_chats(session_id)
    return jsonify({"message": "Session deleted"}), 200