import cv2, time, numpy as np
from app.detector import Detector
from app.tracker import SimpleTracker
from app.rules import ZoneMonitor
from pathlib import Path
import os

# --- Configuration ---
# Use '0' for the live camera feed
VIDEO_SOURCE = 0 
EVIDENCE_FOLDER = "evidence"
RECORDING_DURATION = 5.0 # seconds of video to save after an alert

def draw_zone(frame, poly):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], True, (0,0,255), 2)

def main():
    # Setup folders
    Path(EVIDENCE_FOLDER).mkdir(exist_ok=True)
    
    # Initialize components
    det = Detector()
    tracker = SimpleTracker()
    zone = [(100,100),(500,100),(500,400),(100,400)]
    rules = ZoneMonitor(zone, loiter_seconds=8)
    
    # Camera setup
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video device (Camera).")
        return

    # Get frame properties for video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # Use 30.0 if FPS is 0 (common for webcams)
    
    # Video Writer variables
    video_writer = None
    is_recording = False
    stop_recording_time = 0

    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Error: Failed to receive frame from camera stream or end of file reached.")
            break
            
        t = time.time() - start
        
        # 1. Detection & Tracking
        dets = det.detect(frame)
        dets = [d for d in dets if d[5] == 0] 
        # Tracker returns a list of 4-item tuples: [(oid, bbox_4_tuple, cls, conf), ...]
        tracked = tracker.update(dets) 
        
        # 2. Visualization Loop (FIXED: Iterate directly over the list)
        for oid, bbox, cls, conf in tracked:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID{oid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            
        draw_zone(frame, zone)
        
        # 3. Rules Monitor (takes the list output directly)
        events = rules.update(tracked, t) 
        
        # 4. Video Recording Logic
        for event_index, e in enumerate(events):
            print(f"[{t:.1f}s] {e}")
            cv2.putText(frame, e, (10,30+event_index*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            
            # Check for LOITERING alert to start recording
            if "LOITERING alert" in e and not is_recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(EVIDENCE_FOLDER, f"LOITERING_{timestamp}.mp4")
                
                # Initialize VideoWriter (XVID codec is broadly compatible)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    fps, 
                    (frame_width, frame_height)
                )
                is_recording = True
                stop_recording_time = t + RECORDING_DURATION
                print(f"--- Recording started: {output_path} ---")

        # Write frame if currently recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)
            
            # Check if it's time to stop recording
            if t > stop_recording_time:
                video_writer.release()
                is_recording = False
                print("--- Recording stopped ---")


        # 5. Display Output
        if is_recording:
            cv2.putText(frame, "RECORDING...", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()