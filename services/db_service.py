from models import db, Claim

def create_claim(claim_data: dict) -> str:
    new_claim = Claim(
        claim_type=claim_data.get('claimType'),
        description=claim_data.get('description'),
        policy_number=claim_data.get('policyNumber'),
        incident_date=claim_data.get('incidentDate'),
        contact_phone=claim_data.get('contactPhone', '')
    )
    db.session.add(new_claim)
    db.session.commit()
    return new_claim.id

def get_claim(claim_id: str) -> dict:
    claim = db.session.get(Claim, claim_id)
    return claim.to_dict() if claim else None

def update_claim(claim_id: str, updates: dict):
    claim = db.session.get(Claim, claim_id)
    if claim:
        # Map frontend camelCase to database snake_case
        field_map = {
            'status': 'status',
            'videoUrl': 'video_url',
            'aiConfidence': 'ai_confidence',
            'aiFindings': 'ai_findings',
            'aiRawResponse': 'ai_raw_response',
            'adjusterNotes': 'adjuster_notes',
            'humanReviewed': 'human_reviewed'
        }
        for js_key, db_field in field_map.items():
            if js_key in updates:
                setattr(claim, db_field, updates[js_key])
        db.session.commit()

def list_claims(status_filter: str = None, limit: int = 50) -> list:
    query = Claim.query
    if status_filter and status_filter != 'all':
        query = query.filter(Claim.status == status_filter)
    
    # Order by newest first
    claims = query.order_by(Claim.created_at.desc()).limit(limit).all()
    return [claim.to_dict() for claim in claims]
