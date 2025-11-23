# app/posedetector.py
import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    def __init__(self, mp_holistic=mp.solutions.holistic, mp_drawing=mp.solutions.drawing_utils):
        # Initialize MediaPipe Holistic model (combines pose, face, and hands)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp_drawing
        
        # --- LANDMARKS FOR ROBUSTNESS ---
        # 8: Index Finger Tip (Hand) - The point we measure from
        self.INDEX_FINGER_TIP = 8 
        # 1: Nose (Face) - The anchor point on the face
        self.NOSE_LANDMARK_IDX = 1 
        # 33 & 263: Left and Right Eye inner corner (Face) - Used to measure face size for normalization
        self.LEFT_EYE_INNER = 33
        self.RIGHT_EYE_INNER = 263 
        # --- END LANDMARKS ---

    def _get_pixel_coords(self, landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    def is_smoking_pose(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        frame_height, frame_width, _ = frame.shape
        hand_to_mouth_event = False
        
        if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            
            # 1. GET NORMALIZATION FACTOR (Face Width)
            # Distance between the two inner eye corners is a stable measure of face width.
            l_eye_lm = results.face_landmarks.landmark[self.LEFT_EYE_INNER]
            r_eye_lm = results.face_landmarks.landmark[self.RIGHT_EYE_INNER]
            
            # Convert to pixel coordinates
            l_eye_px, l_eye_py = self._get_pixel_coords(l_eye_lm, frame_width, frame_height)
            r_eye_px, r_eye_py = self._get_pixel_coords(r_eye_lm, frame_width, frame_height)

            # Calculate the face width in pixels
            face_width_px = np.sqrt((l_eye_px - r_eye_px)**2 + (l_eye_py - r_eye_py)**2)
            
            # Exit if face width is zero or too small (e.g., landmark detection failed)
            if face_width_px < 10:
                return False

            # Get the Nose anchor point
            nose_lm = results.face_landmarks.landmark[self.NOSE_LANDMARK_IDX]
            nose_x, nose_y = self._get_pixel_coords(nose_lm, frame_width, frame_height)

            # 2. CHECK HAND PROXIMITY AGAINST NORMALIZED THRESHOLD
            
            hands = []
            if results.right_hand_landmarks:
                hands.append(results.right_hand_landmarks)
            if results.left_hand_landmarks:
                hands.append(results.left_hand_landmarks)
                
            for hand_landmarks in hands:
                tip_lm = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
                hand_x, hand_y = self._get_pixel_coords(tip_lm, frame_width, frame_height)
                
                # Calculate the Hand-to-Nose distance in pixels
                hand_to_nose_distance_px = np.sqrt((nose_x - hand_x)**2 + (nose_y - hand_y)**2)
                
                # Normalize the distance by dividing by the face width
                normalized_distance = hand_to_nose_distance_px / face_width_px
                
                # --- NORMALIZED THRESHOLD ---
                # A good starting point: the hand must be closer than half the face width.
                NORMALIZED_THRESHOLD = 0.5 

                if normalized_distance < NORMALIZED_THRESHOLD: 
                    hand_to_mouth_event = True
                    break

        # --- Draw landmarks for visualization (Keep this for debugging) ---
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)

        return hand_to_mouth_event