import cv2, time, numpy as np
from app.detector import Detector
from app.tracker import SimpleTracker
from app.rules import ZoneMonitor
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# --- Configuration ---
EVIDENCE_FOLDER = "evidence"
RECORDING_DURATION = 5.0 # seconds of video to save after an alert

def draw_zone(frame, poly):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], True, (0,0,255), 2)

# ===============================================
# LOGIC FOR RUNNING LIVE CAMERA/PROCESSING VIDEO
# ===============================================

def run_cctv_logic(video_source):
    # Setup folders
    Path(EVIDENCE_FOLDER).mkdir(exist_ok=True)
    
    # Initialize components
    det = Detector()
    tracker = SimpleTracker()
    zone = [(100,100),(500,100),(500,400),(100,400)]
    rules = ZoneMonitor(zone, loiter_seconds=8)
    
    # Video Capture setup
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source. Check camera index (0) or file path.")
        return

    # Get frame properties for video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # Use 30.0 if FPS is 0 
    
    # Video Writer variables
    video_writer = None
    is_recording = False
    stop_recording_time = 0

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
        
        # 1. Detection & Tracking
        dets = det.detect(frame)
        dets = [d for d in dets if d[5] == 0] 
        # Tracker returns a list of 4-item tuples: [(oid, bbox_4_tuple, cls, conf), ...]
        tracked = tracker.update(dets) 
        
        # 2. Visualization Loop
        for oid, bbox, cls, conf in tracked:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID{oid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            
        draw_zone(frame, zone)
        
        # 3. Rules Monitor
        events = rules.update(tracked, t) 
        
        # 4. Video Recording Logic (only for live feed, or if you want to re-record)
        if video_source == 0:
            for event_index, e in enumerate(events):
                cv2.putText(frame, e, (10,30+event_index*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                
                if "LOITERING alert" in e and not is_recording:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(EVIDENCE_FOLDER, f"LOITERING_{timestamp}.mp4")
                    
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
                
                if t > stop_recording_time:
                    video_writer.release()
                    is_recording = False
                    print("--- Recording stopped ---")

        # 5. Display Output
        if is_recording and video_source == 0:
            cv2.putText(frame, "RECORDING...", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif video_source != 0:
            # Display event log for pre-recorded video analysis
            for event_index, e in enumerate(events):
                cv2.putText(frame, f"[{t:.1f}s] {e}", (10,30+event_index*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)


        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def run_live_footage():
    """Checks live camera footage (index 0)"""
    # Hide the Tkinter window during OpenCV operation
    root.withdraw()
    run_cctv_logic(video_source=0)
    # Show the Tkinter window again when OpenCV closes
    root.deiconify() 

def check_pre_recorded_footage():
    """Checks pre-recorded footage via file dialog"""
    file_path = filedialog.askopenfilename(
        title="Select Pre-Recorded Video",
        initialdir=os.getcwd(), # Start in the current directory
        filetypes=[("Video files", "*.mp4 *.avi")]
    )
    if file_path:
        # Hide the Tkinter window during OpenCV operation
        root.withdraw()
        run_cctv_logic(video_source=file_path)
        # Show the Tkinter window again when OpenCV closes
        root.deiconify()
    else:
        messagebox.showinfo("Cancelled", "No video file selected.")

# ===============================================
# UI SETUP (MAIN APPLICATION ENTRY)
# ===============================================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("CCTV AI Monitor")
    root.geometry("300x150")
    
    # Center the window (optional, but clean)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (300 / 2)
    y = (screen_height / 2) - (150 / 2)
    root.geometry(f'+{int(x)}+{int(y)}')
    
    label = tk.Label(root, text="Select Monitoring Mode", font=("Arial", 12))
    label.pack(pady=15)

    # Live Footage Button
    live_button = tk.Button(
        root, 
        text="Check Live Footage (Webcam)", 
        command=run_live_footage,
        bg="#4CAF50", fg="white", # Green background, white text
        font=("Arial", 10, "bold")
    )
    live_button.pack(pady=5, padx=10, fill='x')

    # Pre-Recorded Footage Button
    pre_recorded_button = tk.Button(
        root, 
        text="Check Pre-Recorded Footage", 
        command=check_pre_recorded_footage,
        bg="#2196F3", fg="white", # Blue background, white text
        font=("Arial", 10, "bold")
    )
    pre_recorded_button.pack(pady=5, padx=10, fill='x')

    root.mainloop()