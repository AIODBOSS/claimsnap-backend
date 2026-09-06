import threading
from flask import Blueprint, request, jsonify, current_app
from services.db_service import create_claim, get_claim, update_claim
from services.storage_service import upload_video, upload_frame, save_temp_file, cleanup_temp_file
from services.ai_service import analyse_claim_video
from utils.video_utils import extract_frames
from config import config

claims_bp = Blueprint('claims', __name__)

def process_claim(app, claim_id, video_path, claim_type):
    # Flask requires the app context to use the database inside a background thread
    with app.app_context():
        try:
            update_claim(claim_id, {'status': 'analysing'})
            frames = extract_frames(video_path, config.FRAMES_TO_EXTRACT)
            
            for i, frame_bytes in enumerate(frames):
                upload_frame(frame_bytes, claim_id, i)
            
            ai_result = analyse_claim_video(frames, claim_type)
            
            update_claim(claim_id, {
                'status': ai_result['recommendation'],
                'aiConfidence': ai_result['confidence'],
                'aiFindings': ai_result['findings'],
                'aiRawResponse': {'labels': ai_result['rawLabels']},
            })
        except Exception as e:
            print(f"Error processing claim {claim_id}: {e}")
            update_claim(claim_id, {
                'status': 'review',
                'adjusterNotes': f'Processing error: {str(e)}',
            })
        finally:
            cleanup_temp_file(video_path)

@claims_bp.route('/claims', methods=['POST'])
def submit_claim():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_bytes = request.files['video'].read()
    
    claim_data = {
        'claimType': request.form.get('claimType', 'other'),
        'description': request.form.get('description', ''),
        'policyNumber': request.form.get('policyNumber', ''),
        'incidentDate': request.form.get('incidentDate', ''),
        'contactPhone': request.form.get('contactPhone', ''),
    }
    
    claim_id = create_claim(claim_data)
    video_url = upload_video(video_bytes, claim_id)
    update_claim(claim_id, {'videoUrl': video_url})
    
    temp_path = save_temp_file(video_bytes, suffix='.webm')
    
    # Run AI analysis in the background
    app = current_app._get_current_object()
    thread = threading.Thread(
        target=process_claim,
        args=(app, claim_id, temp_path, claim_data['claimType']),
        daemon=True,
    )
    thread.start()
    
    return jsonify({
        'claimId': claim_id,
        'status': 'pending',
        'message': 'Claim received. AI analysis in progress.',
    }), 201

@claims_bp.route('/claims/<claim_id>', methods=['GET'])
def get_claim_status(claim_id):
    claim = get_claim(claim_id)
    if not claim:
        return jsonify({'error': 'Claim not found'}), 404
    return jsonify(claim)
