import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

db_url = os.environ.get("DATABASE_URL", "sqlite:///memory_layer.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

model = YOLO("models/best.pt")

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

class ClaimRecord(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    claim_type = db.Column(db.String(50))
    policy_number = db.Column(db.String(100))
    status = db.Column(db.String(50))
    ai_confidence = db.Column(db.Float)
    ai_findings = db.Column(db.Text)
    created_at = db.Column(db.String(100))

with app.app_context():
    import sqlalchemy
    try:
        db.create_all()
    except Exception as e:
        print(f"Skipping DB create: {e}")

@app.route("/api/assess", methods=["POST"])
def assess_claim():
    file_key = "video" if "video" in request.files else "image" if "image" in request.files else None
    if not file_key or "asset_id" not in request.form:
        return jsonify({"error": "Missing media file or asset_id"}), 400
        
    asset_id = request.form["asset_id"]
    media_file = request.files[file_key]
    
    temp_path = f"temp_capture_{asset_id}.webm"
    media_file.save(temp_path)
    
    results = model.predict(source=temp_path, stream=True, conf=0.25)
    highest_conf_per_class = {}
    for frame in results:
        for box in frame.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            if cls_name not in highest_conf_per_class or conf > highest_conf_per_class[cls_name]:
                highest_conf_per_class[cls_name] = conf
                
    detections = [{"class": k, "confidence": v} for k, v in highest_conf_per_class.items()]
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
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
            created_at=datetime.utcnow().isoformat()
        )
        db.session.add(claim)
    else:
        claim.status = final_status
        claim.ai_confidence = highest_conf
        claim.ai_findings = json.dumps(findings_list)
        
    db.session.commit()
    return jsonify({"asset_id": asset_id})

@app.route("/api/claims", methods=["GET"])
def get_all_claims():
    records = ClaimRecord.query.order_by(ClaimRecord.created_at.desc()).all()
    return jsonify([{
        "id": r.id,
        "status": r.status,
        "claimType": r.claim_type,
        "policyNumber": r.policy_number,
        "aiConfidence": round(r.ai_confidence, 1) if r.ai_confidence else None,
        "aiFindings": json.loads(r.ai_findings) if r.ai_findings else [],
        "createdAt": r.created_at
    } for r in records])

@app.route("/api/claims/<claim_id>", methods=["GET"])
def get_claim(claim_id):
    claim = ClaimRecord.query.get(claim_id)
    if not claim:
        return jsonify({"error": "Claim not found"}), 404
        
    return jsonify({
        "id": claim.id,
        "status": claim.status,
        "claimType": claim.claim_type,
        "policyNumber": claim.policy_number,
        "createdAt": claim.created_at,
        "aiConfidence": round(claim.ai_confidence, 1) if claim.ai_confidence else None,
        "aiFindings": json.loads(claim.ai_findings) if claim.ai_findings else []
    })

@app.route("/api/feedback", methods=["POST"])
def adjuster_feedback():
    data = request.json
    asset_id = data.get("asset_id")
    corrected = data.get("corrected_class")
    
    claim = ClaimRecord.query.get(asset_id)
    if claim:
        if "Approved" in corrected:
            claim.status = "approved"
        else:
            claim.status = "rejected"
        
    log = CalibrationLog(asset_id=asset_id, original_class=data.get('original_class', 'Unknown'), corrected_class=corrected)
    history = AssetHistory(asset_id=asset_id, damage_class=corrected, status='overridden')
    db.session.add(log)
    db.session.add(history)
    db.session.commit()
    return jsonify({"message": "Feedback integrated into memory layer"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)

