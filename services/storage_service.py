import os
import tempfile
from werkzeug.utils import secure_filename

# Base directory for local uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def upload_video(video_bytes: bytes, claim_id: str, filename: str = 'claim.webm') -> str:
    claim_dir = os.path.join(UPLOAD_FOLDER, 'claims', claim_id)
    ensure_dir(claim_dir)
    
    safe_filename = secure_filename(filename)
    file_path = os.path.join(claim_dir, safe_filename)
    
    with open(file_path, 'wb') as f:
        f.write(video_bytes)
        
    # Return a relative URL path that Flask can serve
    return f"/uploads/claims/{claim_id}/{safe_filename}"

def upload_frame(frame_bytes: bytes, claim_id: str, frame_index: int) -> str:
    frames_dir = os.path.join(UPLOAD_FOLDER, 'claims', claim_id, 'frames')
    ensure_dir(frames_dir)
    
    filename = f"frame_{frame_index}.jpg"
    file_path = os.path.join(frames_dir, filename)
    
    with open(file_path, 'wb') as f:
        f.write(frame_bytes)
        
    return f"/uploads/claims/{claim_id}/frames/{filename}"

def save_temp_file(content: bytes, suffix: str = '.webm') -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(content)
        return f.name

def cleanup_temp_file(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass
