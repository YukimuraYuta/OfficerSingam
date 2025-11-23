import math

class DrinkingDetector:
    def __init__(self, hand_to_mouth_ratio=0.08):
        """
        hand_to_mouth_ratio = percentage of person height allowed 
        for hand to be considered near mouth.
        Example: 0.08 = 8% of person height.
        """
        self.hand_to_mouth_ratio = hand_to_mouth_ratio

    # ----------------------------------------------------------------------
    # Utility Helpers
    # ----------------------------------------------------------------------
    def center_of_box(self, box):
        """Returns the center (x, y) of a bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def euclidean_distance(self, p1, p2):
        """Returns Euclidean distance between 2 points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # ----------------------------------------------------------------------
    # Core logic
    # ----------------------------------------------------------------------
    def hand_near_mouth(self, hand, mouth, person_box):
        """
        Returns True if hand is close to mouth based on person height.
        """
        if hand is None or mouth is None or person_box is None:
            return False
        
        px1, py1, px2, py2 = person_box
        person_height = py2 - py1
        threshold = person_height * self.hand_to_mouth_ratio

        return self.euclidean_distance(hand, mouth) < threshold

    def is_object_in_hand(self, obj_box, hand_point):
        """
        Checks if hand is inside the bounding box of the object (bottle).
        """
        if obj_box is None or hand_point is None:
            return False
        
        ox1, oy1, ox2, oy2 = obj_box
        hx, hy = hand_point
        return ox1 <= hx <= ox2 and oy1 <= hy <= oy2

    # ----------------------------------------------------------------------
    # PUBLIC FUNCTION CALLED BY cctvprocessor
    # ----------------------------------------------------------------------
    def detect_drinking(self, person_data, bottle_boxes):
        """
        person_data structure expected:
        {
            'bbox': (x1, y1, x2, y2),
            'hand': (hx, hy),
            'mouth': (mx, my)
        }

        bottle_boxes: list of (x1, y1, x2, y2)
        """
        if person_data is None:
            return False
        
        hand = person_data.get("hand")
        mouth = person_data.get("mouth")
        person_box = person_data.get("bbox")

        if not hand or not mouth or not person_box:
            return False

        # Check each detected bottle
        for bottle in bottle_boxes:
            if self.is_object_in_hand(bottle, hand):
                if self.hand_near_mouth(hand, mouth, person_box):
                    return True

        return False
