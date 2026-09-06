from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class Claim(db.Model):
    __tablename__ = 'claims'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    policy_number = db.Column(db.String(100), nullable=False)
    incident_date = db.Column(db.String(20), nullable=False)
    contact_phone = db.Column(db.String(20), nullable=True)
    status = db.Column(db.String(20), default='pending')
    video_url = db.Column(db.String(255), nullable=True)
    
    ai_confidence = db.Column(db.Float, nullable=True)
    ai_findings = db.Column(db.JSON, nullable=True, default=list)
    ai_raw_response = db.Column(db.JSON, nullable=True)
    
    adjuster_notes = db.Column(db.Text, nullable=True)
    human_reviewed = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'claimType': self.claim_type,
            'description': self.description,
            'policyNumber': self.policy_number,
            'incidentDate': self.incident_date,
            'contactPhone': self.contact_phone,
            'status': self.status,
            'videoUrl': self.video_url,
            'aiConfidence': self.ai_confidence,
            'aiFindings': self.ai_findings or [],
            'aiRawResponse': self.ai_raw_response,
            'adjusterNotes': self.adjuster_notes,
            'humanReviewed': self.human_reviewed,
            'createdAt': self.created_at.isoformat() + 'Z' if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() + 'Z' if self.updated_at else None
        }
