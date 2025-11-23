# app/smokedetector.py
from ultralytics import YOLO

class SmokeDetector:
    # --- IMPORTANT CHANGE: Load the local file ---
    # Change the model_name to the local file path: 'app/best.pt'
    def __init__(self, model_name='app/best.pt', device='cpu'):
        # NOTE: If you save the file in your main project folder, change this to 'best.pt'
        self.model = YOLO(model_name) 
        self.model.to(device)

    def detect(self, frame, smoke_class_id=0):
        # NOTE: We assume smoke_class_id=0 based on custom single-class training practices.
        
        # Only run prediction for the specific smoke class
        results = self.model.predict(source=frame, classes=[smoke_class_id], imgsz=640, conf=0.5, verbose=False)
        
        smoke_detected = False
        r = results[0]
        
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            smoke_detected = True
        
        boxes = []
        if smoke_detected:
            # Filter the boxes to ensure they belong to the expected smoke class (ID 0)
            target_boxes = [box for box in r.boxes if int(box.cls[0]) == smoke_class_id]
            boxes = [box.xyxy[0].tolist() for box in target_boxes]

        return smoke_detected, boxes