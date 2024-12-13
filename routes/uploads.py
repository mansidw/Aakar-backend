# routes/uploads.py is a blueprint that defines the routes for uploading files to Llama Cloud.
from flask import Blueprint, jsonify, request
from services.firebase_service import get_firestore_client
from services.llama_service import upload_to_llama_cloud
import os

uploads_bp = Blueprint("uploads", __name__)
db = get_firestore_client()

@uploads_bp.route("/", methods=["POST"])
def upload_files():
    """
    Upload files to Llama Cloud for a project.
    """
    files = request.files.getlist("files")
    print(f"Files: {files}")
    print(f"Project ID: {request.form.get('project_id')}")
    project_id = request.form.get("project_id")

    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400

    # Retrieve project details from Firebase
    project_ref = db.collection("projects").document(project_id)
    project = project_ref.get()
    if not project.exists:
        return jsonify({"error": "Project not found"}), 404

    index_id = project.to_dict().get("index_id")
    print(f"Index ID: {index_id}")
    if not index_id:
        return jsonify({"error": "Llama Cloud index not found for this project"}), 500

    success = []
    errors = []
    print("Uploading files to Llama Cloud...", files)
    for file in files:
        try:
            file_path = f"/tmp/{file.filename}"
            file.save(file_path)
            upload_to_llama_cloud(index_id, file_path, project_id)
            success.append({"file_name": file.filename, "status": "uploaded"})
        except Exception as e:
            errors.append({"file_name": file.filename, "error": str(e)})
            
    return jsonify({"success": success, "errors": errors}), 200
