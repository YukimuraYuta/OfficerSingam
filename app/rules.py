class ZoneMonitor:
    def __init__(self, polygon, loiter_seconds=10):
        self.poly = polygon
        self.loiter_seconds = loiter_seconds
        self.presence = {}

    def _inside(self, cx, cy):
        x,y = cx,cy
        poly = self.poly
        inside = False
        n = len(poly)
        p1x,p1y = poly[0]
        for i in range(1, n+1):
            p2x,p2y = poly[i % n]
            if (y > min(p1y,p2y)) and (y <= max(p1y,p2y)):
                xinters = p1x + (y-p1y)*(p2x-p1x)/(p2y-p1y) if p1y != p2y else p1x
                if x <= xinters:
                    inside = not inside
            p1x,p1y = p2x,p2y
        return inside

    def update(self, tracked_objs, t):
        events=[]
        for oid,bbox,cls,conf in tracked_objs:
            x1,y1,x2,y2 = bbox
            cx,cy = (x1+x2)//2, (y1+y2)//2
            in_zone = self._inside(cx,cy)
            if oid not in self.presence:
                self.presence[oid] = {'in_zone':False,'enter_time':None}
            p = self.presence[oid]
            if in_zone and not p['in_zone']:
                p['in_zone']=True; p['enter_time']=t
                events.append(f"ID {oid} entered zone")
            if p['in_zone'] and (t - (p['enter_time'] or t) > self.loiter_seconds):
                events.append(f"LOITERING alert ID {oid}")
                p['enter_time'] = t  # suppress repeated alerts
            if not in_zone and p['in_zone']:
                dur = t - (p['enter_time'] or t)
                p['in_zone']=False; p['enter_time']=None
                events.append(f"ID {oid} left zone after {dur:.1f}s")
        return events

