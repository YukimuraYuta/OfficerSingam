import cv2, time, numpy as np
from app.detector import Detector
from app.tracker import SimpleTracker
from app.rules import ZoneMonitor
from pathlib import Path

VIDEO_SOURCE = str(Path("data/sample_video.mp4"))

def draw_zone(frame, poly):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], True, (0,0,255), 2)

def main():
    det = Detector()
    tracker = SimpleTracker()
    zone = [(100,100),(500,100),(500,400),(100,400)]
    rules = ZoneMonitor(zone, loiter_seconds=8)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = time.time() - start
        dets = det.detect(frame)
        dets = [d for d in dets if d[5] == 0]  # keep only person class (COCO id 0)
        tracked = tracker.update(dets)
        for oid,bbox,cls,conf in tracked:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID{oid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        draw_zone(frame, zone)
        events = rules.update(tracked, t)
        for i,e in enumerate(events):
            print(f"[{t:.1f}s] {e}")
            cv2.putText(frame, e, (10,30+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

