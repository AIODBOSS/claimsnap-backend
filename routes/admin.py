from flask import Blueprint, request, jsonify
from services.db_service import get_claim, list_claims, update_claim

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/claims', methods=['GET'])
def list_all_claims():
    status = request.args.get('status', 'all')
    claims = list_claims(status_filter=status)
    return jsonify({'claims': claims})

@admin_bp.route('/claims/<claim_id>', methods=['GET'])
def get_claim_detail(claim_id):
    claim = get_claim(claim_id)
    if not claim:
        return jsonify({'error': 'Claim not found'}), 404
    return jsonify(claim)

@admin_bp.route('/claims/<claim_id>/decision', methods=['POST'])
def adjuster_decision(claim_id):
    data = request.get_json() or {}
    decision = data.get('decision')
    notes = data.get('notes', '')
    
    if decision not in ('approved', 'rejected'):
        return jsonify({'error': 'Decision must be "approved" or "rejected"'}), 400
        
    if not get_claim(claim_id):
        return jsonify({'error': 'Claim not found'}), 404
        
    update_claim(claim_id, {
        'status': decision,
        'adjusterNotes': notes,
        'humanReviewed': True,
    })
    return jsonify({'success': True, 'claimId': claim_id, 'newStatus': decision})
