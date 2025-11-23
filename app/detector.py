from ultralytics import YOLO

class Detector:
    def __init__(self, model_name='yolov8n.pt', device='cpu'):
        self.model = YOLO(model_name)
        self.model.to(device)

    def detect(self, frame):
        results = self.model.predict(source=frame, imgsz=640, conf=0.35, verbose=False)
        dets = []
        r = results[0]
        if getattr(r, "boxes", None) is None:
            return dets
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            dets.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
        return dets
