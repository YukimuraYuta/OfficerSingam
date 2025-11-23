# app/cctvprocessor.py
import cv2, time, numpy as np
import os
from pathlib import Path
from tkinter import messagebox
from app.detector import Detector
from app.tracker import SimpleTracker
from app.posedetector import PoseDetector
from app.smokedetector import SmokeDetector

class CCTVProcessor:
    # --- Configuration ---
    EVIDENCE_FOLDER = "evidence"
    RECORDING_DURATION = 5.0 
    SMOKE_CLASS_ID = 0
    SMOKE_WINDOW_SECONDS = 3.0
    TARGET_CLASSES = [0] # Only detect Person (0) for tracking

    # --- STATE MACHINE VARIABLES ---
    POSE_STATE = "NONE"
    POSE_TAKEN_OFF_TIME = 0.0

    def __init__(self):
        Path(self.EVIDENCE_FOLDER).mkdir(exist_ok=True)
        self.det = Detector()
        self.tracker = SimpleTracker()
        self.pose_detector = PoseDetector()
        self.smoke_detector = SmokeDetector()
        self.zone = [(100,100),(500,100),(500,400),(100,400)]

    def _draw_zone(self, frame):
        pts = np.array(self.zone, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0,0,255), 2)

    def run_logic(self, video_source):
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video source. Check camera index (0) or file path.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        video_writer = None
        is_recording = False
        stop_recording_time = 0
        smoking_events = []

        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: 
                if video_source == 0:
                    print("Error: Failed to receive frame from camera stream.")
                else:
                    print("Video playback finished.")
                break
                
            t = time.time() - start
            
            # 1. Detection & Tracking (Person Bounding Boxes)
            dets = self.det.detect(frame)
            dets = [d for d in dets if d[5] in self.TARGET_CLASSES] 
            tracked = self.tracker.update(dets) 
            
            # 2. Pose and Smoke Detection
            # Use a copy of the frame for pose detection to ensure drawing is overlaid correctly
            hand_on_mouth_pose = self.pose_detector.is_smoking_pose(frame.copy()) 
            smoke_detected, smoke_boxes = self.smoke_detector.detect(frame, self.SMOKE_CLASS_ID) 

            # 3. STATE MACHINE LOGIC
            smoking_events.clear() 

            if hand_on_mouth_pose:
                if self.POSE_STATE == "NONE" or self.POSE_STATE == "WAITING_FOR_SMOKE":
                    self.POSE_STATE = "POSE_ACTIVE"
            
            elif self.POSE_STATE == "POSE_ACTIVE":
                # Hand taken off mouth, start timer
                self.POSE_TAKEN_OFF_TIME = t
                self.POSE_STATE = "WAITING_FOR_SMOKE"
                
            elif self.POSE_STATE == "WAITING_FOR_SMOKE":
                if smoke_detected:
                    smoking_events.append("CONFIRMED SMOKING VIOLATION")
                    self.POSE_STATE = "VIOLATION_CONFIRMED"
                elif (t - self.POSE_TAKEN_OFF_TIME) > self.SMOKE_WINDOW_SECONDS:
                    self.POSE_STATE = "NONE"

            elif self.POSE_STATE == "VIOLATION_CONFIRMED":
                smoking_events.append("CONFIRMED SMOKING VIOLATION")
                self.POSE_STATE = "NONE" 
                 
            # 4. Visualization Loop 
            for oid, bbox, cls, conf in tracked:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) # Green for Person
                
                if self.POSE_STATE == "POSE_ACTIVE":
                     color = (255, 165, 0) # Orange for Hand-to-Mouth

                elif smoke_detected or "CONFIRMED SMOKING VIOLATION" in smoking_events:
                     color = (0, 0, 255) # Red for alert
                
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"ID{oid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            # Draw Smoke Bounding Boxes
            for box in smoke_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for Smoke

            self._draw_zone(frame)
            
            # 5. Video Recording Logic 
            if "CONFIRMED SMOKING VIOLATION" in smoking_events and video_source == 0 and not is_recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.EVIDENCE_FOLDER, f"CONFIRMED_SMOKING_{timestamp}.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    fps, 
                    (frame_width, frame_height)
                )
                is_recording = True
                stop_recording_time = t + self.RECORDING_DURATION
                print(f"--- Recording started: {output_path} ---")
                
            # Write frame if currently recording
            if is_recording and video_writer is not None:
                video_writer.write(frame)
                
                if t > stop_recording_time:
                    video_writer.release()
                    is_recording = False
                    print("--- Recording stopped ---")

            # 6. Display Output
            cv2.putText(frame, f"STATE: {self.POSE_STATE}", (frame_width - 200, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i,e in enumerate(smoking_events):
                cv2.putText(frame, e, (10,30+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                
            if is_recording and video_source == 0:
                cv2.putText(frame, "RECORDING...", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("CCTV AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Cleanup
        if video_writer is not None and video_writer.isOpened():
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()