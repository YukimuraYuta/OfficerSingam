from ultralytics import YOLO

class Detector:
    # We will use the yolo detector purely to find the person's bounding box (class 0)
    # The PoseDetector will handle the activity detection.
    def __init__(self, model_name='yolov8n.pt', device='cpu'):
        self.model = YOLO(model_name)
        self.model.to(device)

    def detect(self, frame):
        # We only care about detecting person (class 0) for tracking purposes
        results = self.model.predict(source=frame, classes=[0], imgsz=640, conf=0.35, verbose=False)
        dets = []
        r = results[0]
        if getattr(r, "boxes", None) is None:
            return dets
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Returns 6 elements: (x1, y1, x2, y2, conf, cls)
            dets.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
        return dets