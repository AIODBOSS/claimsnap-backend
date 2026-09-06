import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import config

# Dynamically load the custom UNILAG model if trained, else fallback to base nano model
model_path = 'runs/detect/claimsnap_damage_model/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'yolov8n.pt'

model = YOLO(model_path)

def analyse_frame(frame_bytes: bytes) -> dict:
    # Convert bytes to cv2 image format for YOLO
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLOv8 inference
    results = model(img, verbose=False)
    
    detected_items = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            cls_name = model.names[cls_id]
            detected_items.append({
                'description': cls_name.lower(),
                'score': round(conf, 1)
            })
            
    return {'objects': detected_items}

def analyse_claim_video(frames: list, claim_type: str) -> dict:
    all_objects = []
    for frame_bytes in frames:
        res = analyse_frame(frame_bytes)
        all_objects.extend(res.get('objects', []))
        
    if not all_objects:
        return {'confidence': 0, 'findings': [], 'recommendation': 'review', 'rawLabels': []}
        
    # Group detections by unique object class, keeping the highest confidence score
    damage_indicators = {}
    for obj in all_objects:
        desc = obj['description']
        if desc not in damage_indicators or obj['score'] > damage_indicators[desc]:
            damage_indicators[desc] = obj['score']
            
    # Calculate average confidence of detected items
    scores = list(damage_indicators.values())
    confidence = max(0, min(100, round(sum(scores) / len(scores))))
    
    # Format the top 5 findings for the React frontend
    findings = [
        f"{desc.replace('_', ' ').title()} ({score:.0f}%)" 
        for desc, score in sorted(damage_indicators.items(), key=lambda x: -x[1])
    ][:5]
    
    # Decide outcome based on config thresholds
    if confidence >= config.AUTO_APPROVE_CONFIDENCE:
        recommendation = 'approved'
    elif confidence <= config.AUTO_REJECT_CONFIDENCE:
        recommendation = 'rejected'
    else:
        recommendation = 'review'
        
    return {
        'confidence': confidence,
        'findings': findings,
        'recommendation': recommendation,
        'rawLabels': list(damage_indicators.keys())[:20]
    }
