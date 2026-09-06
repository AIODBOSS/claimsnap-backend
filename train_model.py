from ultralytics import YOLO

def train_model():
    # Load the lightning-fast YOLOv8 Nano base model
    model = YOLO("yolov8n.pt")

    print("Starting YOLOv8 training on 19,565 images...")
    
    # Trigger the training loop
    results = model.train(
        data="dataset/data.yaml",
        epochs=15,          # 15 epochs is enough to generate good thesis graphs without taking days
        imgsz=640,          # Standard image resolution
        batch=8,            # Safe batch size to prevent laptop memory crashes
        plots=True,         # Crucial: Forces generation of Chapter 4 charts
        name="claimsnap_v1" # Output folder name
    )
    
    print("\nTraining complete! Your custom weights and thesis graphs are in: runs/detect/claimsnap_v1")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    train_model()
