import cv2
from config import config

def extract_frames(video_path: str, num_frames: int = None) -> list:
    if num_frames is None:
        num_frames = config.FRAMES_TO_EXTRACT

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        positions = list(range(max(1, total_frames)))
    else:
        step = total_frames // num_frames
        positions = [i * step for i in range(num_frames)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # Encode frame as JPEG bytes
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(encoded.tobytes())

    cap.release()
    return frames
