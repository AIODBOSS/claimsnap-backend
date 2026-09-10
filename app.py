import os
import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db_url = os.environ.get("DATABASE_URL", "sqlite:///memory_layer.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

try:
    from ultralytics import YOLO
    model = YOLO("models/best.pt")
except Exception as e:
    print(f"Warning: YOLO model could not be loaded: {e}")
    model = None

class AssetHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.String(100), nullable=False)
    damage_class = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), default="logged")

class CalibrationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.String(100), nullable=False)
    original_class = db.Column(db.String(50), nullable=False)
    corrected_class = db.Column(db.String(50), nullable=False)

import sqlalchemy as sa

class ClaimRecord(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    claim_type = db.Column(db.String(50))
    policy_number = db.Column(db.String(100))
    status = db.Column(db.String(50))
    ai_status = db.Column(db.String(50), nullable=True)
    ai_confidence = db.Column(db.Float)
    ai_findings = db.Column(db.Text)
    created_at = db.Column(db.String(100))
    admin_corrected_label = db.Column(db.String(50), nullable=True)
    admin_corrected_labels = db.Column(db.Text, nullable=True)
    admin_review_status = db.Column(db.String(30), default="pending")
    admin_decision = db.Column(db.String(30), nullable=True)
    admin_reviewed_at = db.Column(db.String(100), nullable=True)
    model_improvement_consent = db.Column(db.Boolean, default=False, nullable=False)
    consent_timestamp = db.Column(db.String(100), nullable=True)
    consent_version = db.Column(db.String(30), nullable=True)
    training_eligible = db.Column(db.Boolean, default=False, nullable=False)
    video_filename = db.Column(db.String(255), nullable=True)
    upload_received_at = db.Column(db.String(100), nullable=True)

with app.app_context():
    try:
        db.create_all()
        inspector = sa.inspect(db.engine)
        existing_cols = [c['name'] for c in inspector.get_columns('claim_record')]
        with db.engine.connect() as conn:
            if 'admin_corrected_label' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN admin_corrected_label VARCHAR(50);"))
            if 'admin_corrected_labels' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN admin_corrected_labels TEXT;"))
            if 'ai_status' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN ai_status VARCHAR(50);"))
            if 'admin_review_status' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN admin_review_status VARCHAR(30) DEFAULT 'pending';"))
            if 'admin_decision' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN admin_decision VARCHAR(30);"))
            if 'admin_reviewed_at' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN admin_reviewed_at VARCHAR(100);"))
            if 'model_improvement_consent' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN model_improvement_consent BOOLEAN DEFAULT FALSE;"))
            if 'consent_timestamp' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN consent_timestamp VARCHAR(100);"))
            if 'consent_version' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN consent_version VARCHAR(30);"))
            if 'training_eligible' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN training_eligible BOOLEAN DEFAULT FALSE;"))
            if 'video_filename' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN video_filename VARCHAR(255);"))
            if 'upload_received_at' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN upload_received_at VARCHAR(100);"))
            conn.commit()
    except Exception as e:
        db.session.rollback()
        print(f"DB Init/Migration Note: {e}")

TRAINING_CONSENT_VERSION = "1.0"

def update_training_eligibility(claim):
    claim.training_eligible = bool(claim.model_improvement_consent and claim.admin_review_status == "reviewed")

@app.route("/api/assess", methods=["POST"])
def assess_claim():
    try:
        file_key = "video" if "video" in request.files else "image" if "image" in request.files else None
        if not file_key or "asset_id" not in request.form:
            return jsonify({"error": "Missing media file or asset_id"}), 400
            
        asset_id = request.form["asset_id"]
        media_file = request.files[file_key]
        
        filename = f"claim_{asset_id}_{int(datetime.utcnow().timestamp())}.webm"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        media_file.save(save_path)
        upload_received_at = datetime.utcnow().isoformat()
        model_improvement_consent = request.form.get("modelImprovementConsent", "false").lower() == "true"
        consent_timestamp = upload_received_at if model_improvement_consent else None
        
        detections = []
        if model:
            results = model.predict(source=save_path, stream=True, conf=0.25)
            highest_conf_per_class = {}
            for frame in results:
                for box in frame.boxes:
                    cls_name = model.names[int(box.cls)]
                    conf = float(box.conf)
                    if cls_name not in highest_conf_per_class or conf > highest_conf_per_class[cls_name]:
                        highest_conf_per_class[cls_name] = conf
            detections = [{"class": k, "confidence": v} for k, v in highest_conf_per_class.items()]
        else:
            return jsonify({
                "error": "AI model is unavailable. Claim assessment cannot be completed."
            }), 503
            
        flags = []
        history = AssetHistory.query.filter_by(asset_id=asset_id).all()
        for past in history:
            for det in detections:
                if det["class"] == past.damage_class and past.status == "repaired":
                    flags.append(f"Flag: {past.damage_class} was previously repaired on this asset.")

        if not detections:
            final_status = "rejected"
        elif flags:
            final_status = "review"
        else:
            final_status = "approved"
        
        highest_conf = max([d["confidence"] for d in detections]) * 100 if detections else 0
        findings_list = [f"{d['class'].capitalize()}" for d in detections] if detections else ["No damage detected"]
        
        claim = ClaimRecord.query.get(asset_id)
        if not claim:
            claim = ClaimRecord(
                id=asset_id,
                claim_type=request.form.get("claimType", "Unknown"),
                policy_number=asset_id,
                status=final_status,
                ai_status=final_status,
                ai_confidence=highest_conf,
                ai_findings=json.dumps(findings_list),
                created_at=datetime.utcnow().isoformat(),
                model_improvement_consent=model_improvement_consent,
                consent_timestamp=consent_timestamp,
                consent_version=TRAINING_CONSENT_VERSION if model_improvement_consent else None,
                training_eligible=False,
                video_filename=filename,
                upload_received_at=upload_received_at
            )
            db.session.add(claim)
        else:
            claim.status = final_status
            claim.ai_status = final_status
            claim.ai_confidence = highest_conf
            claim.ai_findings = json.dumps(findings_list)
            claim.model_improvement_consent = model_improvement_consent
            claim.consent_timestamp = consent_timestamp
            claim.consent_version = TRAINING_CONSENT_VERSION if model_improvement_consent else None
            claim.training_eligible = False
            claim.video_filename = filename
            claim.upload_received_at = upload_received_at
            
        db.session.commit()
        return jsonify({"asset_id": asset_id})
    except Exception as e:
        db.session.rollback()
        print(f"Error in assess_claim: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/media/<filename>", methods=["GET"])
def get_media(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/claims", methods=["GET"])
def get_all_claims():
    records = ClaimRecord.query.order_by(ClaimRecord.created_at.desc()).all()
    return jsonify([{
        "id": r.id,
        "status": r.status,
        "aiStatus": r.ai_status or r.status,
        "claimType": r.claim_type,
        "policyNumber": r.policy_number,
        "aiConfidence": round(r.ai_confidence, 1) if r.ai_confidence is not None else None,
        "aiFindings": json.loads(r.ai_findings) if r.ai_findings else [],
        "createdAt": r.created_at,
        "adminCorrectedLabel": r.admin_corrected_label,
        "adminCorrectedLabels": json.loads(r.admin_corrected_labels) if r.admin_corrected_labels else [],
        "adminReviewStatus": r.admin_review_status or "pending",
        "adminDecision": r.admin_decision,
        "adminReviewedAt": r.admin_reviewed_at,
        "modelImprovementConsent": bool(r.model_improvement_consent),
        "consentTimestamp": r.consent_timestamp,
        "consentVersion": r.consent_version,
        "trainingEligible": bool(r.training_eligible),
        "videoUrl": f"/api/media/{r.video_filename}" if r.video_filename else None
    } for r in records])

@app.route("/api/claims/<string:claim_id>", methods=["GET"])
def get_claim_by_id(claim_id):
    r = ClaimRecord.query.get(claim_id)
    if not r:
        return jsonify({"error": "Claim not found"}), 404
    return jsonify({
        "id": r.id,
        "status": r.status,
        "aiStatus": r.ai_status or r.status,
        "claimType": r.claim_type,
        "policyNumber": r.policy_number,
        "aiConfidence": round(r.ai_confidence, 1) if r.ai_confidence is not None else None,
        "aiFindings": json.loads(r.ai_findings) if r.ai_findings else [],
        "createdAt": r.created_at,
        "adminCorrectedLabel": r.admin_corrected_label,
        "adminCorrectedLabels": json.loads(r.admin_corrected_labels) if r.admin_corrected_labels else [],
        "adminReviewStatus": r.admin_review_status or "pending",
        "adminDecision": r.admin_decision,
        "adminReviewedAt": r.admin_reviewed_at,
        "modelImprovementConsent": bool(r.model_improvement_consent),
        "consentTimestamp": r.consent_timestamp,
        "consentVersion": r.consent_version,
        "trainingEligible": bool(r.training_eligible),
        "videoUrl": f"/api/media/{r.video_filename}" if r.video_filename else None
    })

@app.route('/api/admin/override/<string:claim_id>', methods=['POST'])
def admin_override(claim_id):
    claim = ClaimRecord.query.get(claim_id)
    if not claim:
        return jsonify({"error": "Claim not found"}), 404

    data = request.json or {}

    decision = data.get('decision')
    corrected_labels = data.get('corrected_labels')

    if decision not in ('approved', 'rejected'):
        return jsonify({"error": "Decision must be 'approved' or 'rejected'"}), 400

    if corrected_labels is None:
        corrected_labels = data.get('corrected_label')

    if isinstance(corrected_labels, str):
        corrected_labels = [corrected_labels] if corrected_labels.strip() else []

    if not isinstance(corrected_labels, list):
        return jsonify({"error": "corrected_labels must be a list"}), 400

    taxonomy = {
        'vehicle_dent', 'bumper_damage', 'glass_damage',
        'light_damage', 'mirror_damage', 'runningboard_damage',
        'crack', 'corrosion', 'mold', 'peeling', 'spalling'
    }

    cleaned_labels = []
    for label in corrected_labels:
        if not isinstance(label, str):
            continue
        label = label.strip()
        if label and label in taxonomy and label not in cleaned_labels:
            cleaned_labels.append(label)

    claim.status = decision
    claim.admin_corrected_labels = json.dumps(cleaned_labels)
    claim.admin_corrected_label = cleaned_labels[0] if cleaned_labels else None
    claim.admin_decision = decision
    claim.admin_review_status = "reviewed"
    claim.admin_reviewed_at = datetime.utcnow().isoformat()
    update_training_eligibility(claim)

    db.session.commit()

    return jsonify({
        'message': 'Administrator review saved successfully',
        'claim_id': claim.id,
        'corrected_labels': cleaned_labels,
        'decision': claim.admin_decision,
        'admin_review_status': claim.admin_review_status
    }), 200


@app.route('/api/admin/export/training-feedback', methods=['GET'])
def export_training_feedback():
    """Export consented, administrator-reviewed training-feedback metadata.

    Prototype endpoint: add real administrator authentication before production use.
    """
    import io
    from flask import make_response

    records = ClaimRecord.query.filter(
        ClaimRecord.model_improvement_consent.is_(True),
        ClaimRecord.admin_review_status == "reviewed"
    ).order_by(ClaimRecord.created_at.asc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "claim_id", "claim_type", "policy_number", "created_at",
        "ai_status", "ai_confidence", "ai_findings",
        "admin_review_status", "admin_decision", "admin_corrected_labels",
        "admin_reviewed_at", "model_improvement_consent", "consent_timestamp",
        "consent_version", "training_eligible", "video_filename"
    ])

    for r in records:
        writer.writerow([
            r.id, r.claim_type, r.policy_number, r.created_at,
            r.ai_status or r.status, r.ai_confidence, r.ai_findings or "[]",
            r.admin_review_status or "pending", r.admin_decision or "",
            r.admin_corrected_labels or "[]", r.admin_reviewed_at or "",
            bool(r.model_improvement_consent), r.consent_timestamp or "",
            r.consent_version or "", bool(r.training_eligible), r.video_filename or ""
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=claimsnap_training_feedback.csv"
    return response

@app.route('/api/admin/export/training-package', methods=['GET'])
def export_training_package():
    """Export consented, reviewed media plus a CSV manifest for future model training."""
    import io
    import zipfile
    from flask import make_response

    records = ClaimRecord.query.filter(
        ClaimRecord.model_improvement_consent.is_(True),
        ClaimRecord.admin_review_status == "reviewed",
        ClaimRecord.training_eligible.is_(True)
    ).order_by(ClaimRecord.created_at.asc()).all()

    memory = io.BytesIO()
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow([
        "claim_id", "video_filename", "claim_type", "ai_status",
        "ai_findings", "admin_decision", "admin_corrected_labels",
        "consent_version", "training_eligible"
    ])

    added_media = 0
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in records:
            writer.writerow([
                r.id, r.video_filename or "", r.claim_type,
                r.ai_status or r.status, r.ai_findings or "[]",
                r.admin_decision or "", r.admin_corrected_labels or "[]",
                r.consent_version or "", bool(r.training_eligible)
            ])
            if r.video_filename:
                media_path = os.path.join(UPLOAD_FOLDER, r.video_filename)
                if os.path.isfile(media_path):
                    zf.write(media_path, arcname=f"media/{r.video_filename}")
                    added_media += 1
        zf.writestr("training_feedback.csv", manifest.getvalue())
        zf.writestr("README.txt", "Contains only records with explicit model-improvement consent and completed administrator review.\n")

    memory.seek(0)
    response = make_response(memory.read())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = "attachment; filename=claimsnap_training_package.zip"
    response.headers["X-Exported-Records"] = str(len(records))
    response.headers["X-Exported-Media"] = str(added_media)
    return response

if __name__ == "__main__":
    app.run(port=5000, debug=True)

