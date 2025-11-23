# tracker.py
import math

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}

    def update(self, detections):
        updated_objects = {}
        for det in detections:
            # Unpack the full 6-item detection tuple (x1, y1, x2, y2, conf, cls)
            x1, y1, x2, y2, conf, cls = det 
            assigned_id = None

            # Try matching with existing objects
            for obj_id, det_old in self.objects.items():
                # Unpack all 6 elements from the old detection (using '_' for ignored values)
                ox1, oy1, ox2, oy2, _, _ = det_old
                
                # Calculate distance between centers (approximation using top-left corner here)
                distance = math.hypot((x1 - ox1), (y1 - oy1))

                if distance < 30:
                    assigned_id = obj_id
                    break

            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1

            # Store the full 6-item detection tuple internally
            updated_objects[assigned_id] = det

        self.objects = updated_objects
        
        # FIX: Convert internal dict to the required list format for main.py/rules.py
        # Format: [(oid, (x1, y1, x2, y2), cls, conf), ...]
        output_tracked = []
        for oid, det in self.objects.items():
            x1, y1, x2, y2, conf, cls = det
            bbox = (x1, y1, x2, y2)
            # The original main/rules code expects a 4-item tuple per object:
            # (oid, bbox_tuple, cls, conf)
            output_tracked.append((oid, bbox, cls, conf))
            
        return output_tracked