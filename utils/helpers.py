# utils/helpers.py
import os
import uuid

def generate_unique_filename(filename):
    """Generate a unique filename to prevent collisions."""
    unique_id = uuid.uuid4().hex
    name, ext = os.path.splitext(filename)
    return f"{name}_{unique_id}{ext}"
