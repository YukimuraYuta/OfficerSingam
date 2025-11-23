import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
# Import the new processing class
from app.cctvprocessor import CCTVProcessor


def run_live_footage():
    """Checks live camera footage (index 0)"""
    root.withdraw()
    processor = CCTVProcessor()
    processor.run_logic(video_source=0)
    root.deiconify() 

def check_pre_recorded_footage():
    """Checks pre-recorded footage via file dialog"""
    file_path = filedialog.askopenfilename(
        title="Select Pre-Recorded Video",
        initialdir=os.getcwd(),
        filetypes=[("Video files", "*.mp4 *.avi")]
    )
    if file_path:
        root.withdraw()
        processor = CCTVProcessor()
        processor.run_logic(video_source=file_path)
        root.deiconify()
    else:
        messagebox.showinfo("Cancelled", "No video file selected.")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("CCTV AI Monitor")
    root.geometry("300x150")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (300 / 2)
    y = (screen_height / 2) - (150 / 2)
    root.geometry(f'+{int(x)}+{int(y)}')
    
    label = tk.Label(root, text="Select Monitoring Mode", font=("Arial", 12))
    label.pack(pady=15)

    live_button = tk.Button(
        root, 
        text="Check Live Footage (Webcam)", 
        command=run_live_footage,
        bg="#4CAF50", fg="white", 
        font=("Arial", 10, "bold")
    )
    live_button.pack(pady=5, padx=10, fill='x')

    pre_recorded_button = tk.Button(
        root, 
        text="Check Pre-Recorded Footage", 
        command=check_pre_recorded_footage,
        bg="#2196F3", fg="white", 
        font=("Arial", 10, "bold")
    )
    pre_recorded_button.pack(pady=5, padx=10, fill='x')

    root.mainloop()