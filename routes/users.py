from flask import Blueprint, jsonify, request
from services.firebase_service import get_firestore_client
import uuid
from datetime import datetime

users_bp = Blueprint("users", __name__)
db = get_firestore_client()

def get_timestamp():
    return datetime.utcnow().isoformat()

@users_bp.route("/", methods=["POST"])
def create_user():
    data = request.json
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "name": data.get("name"),
        "email": data.get("email"),
        "password": data.get("password"),
        "created_at": get_timestamp(),
        "updated_at": get_timestamp(),
    }
    db.collection("users").document(user_id).set(user_data)
    return jsonify(user_data), 201

# login user
@users_bp.route("/login", methods=["POST"])
def login_user():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    user_ref = db.collection("users")
    users = [doc.to_dict() for doc in user_ref.stream()]
    for user in users:
        if user.get("email") == email and user.get("password") == password:
            return jsonify(user), 200
    return jsonify({"error": "User not found"}), 404

@users_bp.route("/", methods=["GET"])
def list_users():
    users_ref = db.collection("users")
    users = [doc.to_dict() for doc in users_ref.stream()]
    return jsonify(users), 200

@users_bp.route("/<user_id>", methods=["GET"])
def get_user(user_id):
    user_ref = db.collection("users").document(user_id)
    user = user_ref.get()
    if user.exists:
        return jsonify(user.to_dict()), 200
    return jsonify({"error": "User not found"}), 404

@users_bp.route("/<user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.json
    user_ref = db.collection("users").document(user_id)
    user = user_ref.get()
    if user.exists:
        user_ref.update({
            "name": data.get("name"),
            "email": data.get("email"),
            "updated_at": get_timestamp(),
        })
        return jsonify({"message": "User updated successfully"}), 200
    return jsonify({"error": "User not found"}), 404

@users_bp.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    user_ref = db.collection("users").document(user_id)
    user = user_ref.get()
    if user.exists:
        user_ref.delete()
        return jsonify({"message": "User deleted successfully"}), 200
    return jsonify({"error": "User not found"}), 404
