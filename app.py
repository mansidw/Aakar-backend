# app.py
from flask import Flask, jsonify
from routes.users import users_bp
from routes.projects import projects_bp
from routes.uploads import uploads_bp
from routes.reports import reports_bp
from routes.chats import chats_bp
from flask_swagger_ui import get_swaggerui_blueprint
import logging
from flask_socketio import SocketIO
from flask_cors import CORS
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 
CORS(app)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Register Blueprints
app.register_blueprint(users_bp, url_prefix="/users")
app.register_blueprint(projects_bp, url_prefix="/projects")
app.register_blueprint(uploads_bp, url_prefix="/uploads")
app.register_blueprint(reports_bp, url_prefix="/reports")
app.register_blueprint(chats_bp, url_prefix="/chats")

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  # Ensure you have a swagger.yaml
swaggerui_bp = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "AI Report Generator API"}
)
app.register_blueprint(swaggerui_bp, url_prefix=SWAGGER_URL)

@app.route("/", methods=["GET"])
def home():
    return {"message": "AI Report Generator is running"}, 200

# Error Handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    logger.error(f"Not Found: {error}")
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    socketio.run(app, debug=True)
