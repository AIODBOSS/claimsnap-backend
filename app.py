import os
import json
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
    ai_confidence = db.Column(db.Float)
    ai_findings = db.Column(db.Text)
    created_at = db.Column(db.String(100))
    admin_corrected_label = db.Column(db.String(50), nullable=True)
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
            if 'video_filename' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN video_filename VARCHAR(255);"))
            if 'upload_received_at' not in existing_cols:
                conn.execute(db.text("ALTER TABLE claim_record ADD COLUMN upload_received_at VARCHAR(100);"))
            conn.commit()
    except Exception as e:
        db.session.rollback()
        print(f"DB Init/Migration Note: {e}")

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
                ai_confidence=highest_conf,
                ai_findings=json.dumps(findings_list),
                created_at=datetime.utcnow().isoformat(),
                video_filename=filename,
                upload_received_at=upload_received_at
            )
            db.session.add(claim)
        else:
            claim.status = final_status
            claim.ai_confidence = highest_conf
            claim.ai_findings = json.dumps(findings_list)
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
        "claimType": r.claim_type,
        "policyNumber": r.policy_number,
        "aiConfidence": round(r.ai_confidence, 1) if r.ai_confidence is not None else None,
        "aiFindings": json.loads(r.ai_findings) if r.ai_findings else [],
        "createdAt": r.created_at,
        "adminCorrectedLabel": r.admin_corrected_label,
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
        "claimType": r.claim_type,
        "policyNumber": r.policy_number,
        "aiConfidence": round(r.ai_confidence, 1) if r.ai_confidence is not None else None,
        "aiFindings": json.loads(r.ai_findings) if r.ai_findings else [],
        "createdAt": r.created_at,
        "adminCorrectedLabel": r.admin_corrected_label,
        "videoUrl": f"/api/media/{r.video_filename}" if r.video_filename else None
    })

@app.route('/api/admin/override/<string:claim_id>', methods=['POST'])
def admin_override(claim_id):
    claim = ClaimRecord.query.get(claim_id)
    if not claim:
        return jsonify({"error": "Claim not found"}), 404
    data = request.json or {}
    claim.status = data.get('status', claim.status)
    claim.admin_corrected_label = data.get('corrected_label')
    db.session.commit()
    return jsonify({
        'message': 'Override saved successfully',
        'claim_id': claim.id,
        'corrected_label': claim.admin_corrected_label,
        'status': claim.status
    }), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)

