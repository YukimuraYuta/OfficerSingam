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
        
        # --- LANDMARKS ---
        self.INDEX_FINGER_TIP = 8 
        self.NOSE_LANDMARK_IDX = 1
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
            
            # 1. Face width for normalization
            l_eye_lm = results.face_landmarks.landmark[self.LEFT_EYE_INNER]
            r_eye_lm = results.face_landmarks.landmark[self.RIGHT_EYE_INNER]
            
            l_eye_px, l_eye_py = self._get_pixel_coords(l_eye_lm, frame_width, frame_height)
            r_eye_px, r_eye_py = self._get_pixel_coords(r_eye_lm, frame_width, frame_height)

            face_width_px = np.sqrt((l_eye_px - r_eye_px)**2 + (l_eye_py - r_eye_py)**2)
            if face_width_px < 10:
                return False

            nose_lm = results.face_landmarks.landmark[self.NOSE_LANDMARK_IDX]
            nose_x, nose_y = self._get_pixel_coords(nose_lm, frame_width, frame_height)

            hands = []
            if results.right_hand_landmarks:
                hands.append(results.right_hand_landmarks)
            if results.left_hand_landmarks:
                hands.append(results.left_hand_landmarks)
                
            for hand_landmarks in hands:
                tip_lm = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
                hand_x, hand_y = self._get_pixel_coords(tip_lm, frame_width, frame_height)
                
                hand_to_nose_distance_px = np.sqrt((nose_x - hand_x)**2 + (nose_y - hand_y)**2)
                normalized_distance = hand_to_nose_distance_px / face_width_px
                
                if normalized_distance < 0.5:
                    hand_to_mouth_event = True
                    break

        # Draw landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)

        return hand_to_mouth_event


    # ----------------------------------------------------------
    # NEW METHOD ADDED FOR DRINKING DETECTION (DO NOT MODIFY ABOVE)
    # ----------------------------------------------------------
    def get_person_points(self, frame, bbox):
        """
        Returns hand and mouth coordinates for drinking detection.
        bbox = person bounding box (x1, y1, x2, y2)
        """

        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        hh, ww, _ = person_crop.shape
        img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # No detected landmarks?
        if not results.face_landmarks:
            return None

        # Mouth anchor = nose landmark
        mouth_lm = results.face_landmarks.landmark[self.NOSE_LANDMARK_IDX]
        mouth_x = int(mouth_lm.x * ww) + x1
        mouth_y = int(mouth_lm.y * hh) + y1
        mouth = (mouth_x, mouth_y)

        # Select ANY available hand
        hand = None
        hand_landmarks = None

        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks
        elif results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks

        if hand_landmarks:
            tip_lm = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
            hand_x = int(tip_lm.x * ww) + x1
            hand_y = int(tip_lm.y * hh) + y1
            hand = (hand_x, hand_y)

        return {
            "hand": hand,
            "mouth": mouth
        }
