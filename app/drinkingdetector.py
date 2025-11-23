# app/drinking_detector.py
import math

class DrinkingDetector:
    
    def _distance(self, p1, p2):
        if p1 is None or p2 is None:
            return 9999
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _center(self, box):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def detect_drinking(self, person_data, bottle_boxes):
        """
        person_data must contain: 
           hand, mouth  (points from pose detector)
        bottle_boxes: list of bottle bbox detections
        """

        if "hand" not in person_data or "mouth" not in person_data:
            return False
        
        hand = person_data["hand"]
        mouth = person_data["mouth"]

        if hand is None or mouth is None:
            return False

        # Hand near mouth?
        dist_hand_mouth = self._distance(hand, mouth)
        if dist_hand_mouth > 80:   # threshold (px)
            return False

        # Bottle near hand?
        for bottle in bottle_boxes:
            b_center = self._center(bottle)
            if b_center is not None:
                dist_hand_bottle = self._distance(hand, b_center)
                if dist_hand_bottle < 120:
                    return True

        return False
