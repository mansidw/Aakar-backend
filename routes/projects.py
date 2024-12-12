from flask import Blueprint, jsonify, request
from services.firebase_service import get_firestore_client
from services.llama_service import create_llama_index, list_files_in_llama_cloud
import uuid
from datetime import datetime

projects_bp = Blueprint("projects", __name__)
db = get_firestore_client()

def get_timestamp():
    return datetime.utcnow().isoformat()

@projects_bp.route("/", methods=["POST"])
def create_project():
    data = request.json
    project_id = str(uuid.uuid4())
    
    # Create the Llama Cloud index
    try:
        index = create_llama_index(project_id)
        index_id = index.id
    except Exception as e:
        return jsonify({"error": f"Failed to create Llama Cloud index: {str(e)}"}), 500

    # Save project metadata to Firebase
    project_data = {
        "project_id": project_id,
        "user_id": data.get("user_id"),
        "project_name": data.get("project_name"),
        "index_id": index_id,
        "created_at": get_timestamp(),
        "updated_at": get_timestamp(),
    }
    db.collection("projects").document(project_id).set(project_data)

    return jsonify(project_data), 201

@projects_bp.route("/", methods=["GET"])
def list_projects():
    projects_ref = db.collection("projects")
    projects = [doc.to_dict() for doc in projects_ref.stream()]
    return jsonify(projects), 200

@projects_bp.route("/<project_id>", methods=["GET"])
def get_project(project_id):
    project_ref = db.collection("projects").document(project_id)
    project = project_ref.get()
    if project.exists:
        return jsonify(project.to_dict()), 200
    return jsonify({"error": "Project not found"}), 404

@projects_bp.route("/<project_id>/files", methods=["GET"])
def list_project_files(project_id):
    files = list_files_in_llama_cloud(project_id)
    return jsonify({"project_id": project_id, "files": files}), 200
