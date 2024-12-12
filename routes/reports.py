# routes/reports.py - API routes for generating reports
from flask import Blueprint, jsonify, request, send_file
from services.llama_service import initialize_agent
from services.report_agent import run_agent
from services.firebase_service import get_firestore_client
from services.report_formatter import generate_pdf, generate_docx, generate_html, generate_markdown
from services.chart_service import ChartService
from services.image_service import ImageService
from services.chat_service import create_chat_session_if_not_exists
import logging
import asyncio
import os

reports_bp = Blueprint("reports", __name__)
db = get_firestore_client()
logger = logging.getLogger(__name__)

@reports_bp.route("/generate", methods=["POST"])
def generate_report_api():
    """
    API to generate a report based on user query.
    """
    data = request.json
    project_id = data.get("project_id")
    query = data.get("query")
    user_id = data.get("user_id")
    report_format = data.get("format", "MARKDOWN").upper()
    session_id = data.get("session_id", None)

    logger.info(f"Received report generation request for project_id: {project_id}, user_id: {user_id}")

    if not project_id or not query or not user_id:
        logger.error("Missing required fields: project_id, user_id, or query.")
        return jsonify({"error": "Project ID, User ID, and query are required"}), 400

    # Ensure session_id
    session_id = create_chat_session_if_not_exists(user_id, project_id, session_id)

    # Run agent to get report
    try:
        result = asyncio.run(run_agent(input_query=query, index_name=f"project_{project_id}_index"))
    except Exception as e:
        logger.error(f"Failed to run agent: {str(e)}")
        return jsonify({"error": f"Failed to run agent: {str(e)}"}), 500

    # result is a ReportOutput instance
    # result_dict = {"blocks": [...]} from run_agent function
    # Convert this result into final file based on report_format
    blocks = result.blocks

    # Generate charts if ChartBlocks found
    chart_service = ChartService()
    charts_data = []
    for block in blocks:
        if block.__class__.__name__ == "ChartBlock":
            charts_data.append({
                "type": block.type,
                "title": block.title,
                "data": block.data
            })
    chart_paths = chart_service.generate_charts(charts_data)

    # If there are any ImageBlocks, handle them - currently assume images come from text?
    # If LLM produces image references, you'd retrieve and process them similarly
    image_service = ImageService()
    image_paths = []  # If we have image logic from blocks, handle here

    # Convert blocks to a suitable structure for formatting
    # Extract text content
    text_content = ""
    tables = []
    # We'll store Chart and Image references separately
    for block in blocks:
        if block.__class__.__name__ == "TextBlock":
            text_content += block.text + "\n\n"
        elif block.__class__.__name__ == "TableBlock":
            tables.append({
                "title": block.caption,
                "columns": block.col_names,
                "rows": block.rows
            })
        # Chart and images are already handled

    # Format report
    if report_format == "PDF":
        report_path = generate_pdf(text_content, tables, chart_paths, image_paths)
    elif report_format == "DOCX":
        report_path = generate_docx(text_content, tables, chart_paths, image_paths)
    elif report_format == "HTML":
        report_path = generate_html(text_content, tables, chart_paths, image_paths)
    else:
        # Default or MARKDOWN
        report_path = generate_markdown(text_content, tables, chart_paths, image_paths)

    # Return the file
    if report_format in ["PDF", "DOCX", "HTML", "MARKDOWN"]:
        # Determine MIME type
        mime_types = {
            "PDF": "application/pdf",
            "DOCX": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "HTML": "text/html",
            "MARKDOWN": "text/markdown"
        }
        mime_type = mime_types.get(report_format, "text/markdown")
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"report.{report_format.lower()}",
            mimetype=mime_type
        )
    else:
        # Just return the markdown content directly
        with open(report_path, "r") as f:
            content = f.read()
        return jsonify({"report": content}), 200