import math
import numpy as np
from PIL import Image


class RacingSimulator:
    WALL = (15, 15, 15)
    TRACK = (35, 35, 35)
    GRASS = (34, 177, 76)

    MAX_SPEED = 300.0
    ACCELERATION = 150.0
    FRICTION = 50.0
    TURN_BASE = 3.0
    TURN_FACTOR = 0.3

    SHORT_RANGE = 200.0
    LONG_RANGE = 900.0
    REF_DIST = 50.0

    SHORT_OFFSETS = [
        -math.pi/2, -5*math.pi/12, -math.pi/3, -math.pi/4,
        -math.pi/6, -math.pi/12, 0.0, math.pi/12,
        math.pi/6, math.pi/4, math.pi/3, 5*math.pi/12, math.pi/2
    ]
    LONG_OFFSETS = [-math.pi/6, -math.pi/12, 0.0, math.pi/12, math.pi/6]

    CHECKPOINTS = [
        ((450, 35),  (450, 150)),
        ((719, 260), (850, 260)),
        ((850, 665), (723, 665)),
        ((523, 482), (625, 517)),
        ((409, 438), (295, 413)),
        ((160, 730), (220, 815)),
        ((138, 600), (49,  600)),
        ((138, 205), (49,  205)),
    ]
    TOTAL_LAPS = 3

    def __init__(self, track_path):
        img = Image.open(track_path).convert('RGB')
        self.pixels = np.array(img, dtype=np.uint8)
        self.W = img.width
        self.H = img.height
        self.reset()

    def reset(self):
        self.x = 430.0
        self.y = 92.0
        self.angle = 0.0
        self.speed = 0.0
        self.current_lap = -1
        self.next_cp = 0
        self.cp_crossed = [False] * len(self.CHECKPOINTS)
        self.lap_time = 0.0
        self.best_lap = float('inf')
        self.lap_times = []
        self.finished = False
        self.wall_hits = 0
        self.steps = 0

    def _pixel(self, x, y):
        px, py = int(x), int(y)
        if 0 <= px < self.W and 0 <= py < self.H:
            r, g, b = self.pixels[py, px]
            return (int(r), int(g), int(b))
        return None

    def _is_wall(self, c):
        return c is None or c == self.WALL

    def _friction_mult(self, c):
        if c is None or c == self.WALL:
            return 999.0
        if c == self.GRASS:
            return 3.0
        return 1.0

    def _cast_ray(self, ox, oy, ra, max_d):
        cos_a, sin_a = math.cos(ra), math.sin(ra)
        d = 0.0
        while d < max_d:
            x = ox + cos_a * d
            y = oy + sin_a * d
            px, py = int(x), int(y)
            if px < 0 or px >= self.W or py < 0 or py >= self.H:
                return d, x, y
            r, g, b = self.pixels[py, px]
            if (int(r), int(g), int(b)) == self.WALL:
                return d, x, y
            d += 2.0
        hx = ox + cos_a * max_d
        hy = oy + sin_a * max_d
        return max_d, hx, hy

    def _seg_cross(self, ax, ay, bx, by, cx, cy, dx, dy):
        denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
        if abs(denom) < 1e-6:
            return False
        t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
        u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    def get_state(self):
        state = [
            self.speed / self.MAX_SPEED,
            math.sin(self.angle),
            math.cos(self.angle),
            self.x / self.W,
            self.y / self.H,
        ]
        for off in self.SHORT_OFFSETS:
            d, _, _ = self._cast_ray(self.x, self.y, self.angle + off, self.SHORT_RANGE)
            danger = min(1.0, 1.0 / ((d / self.REF_DIST) + 0.1))
            state.append(danger)
        for off in self.LONG_OFFSETS:
            d, _, _ = self._cast_ray(self.x, self.y, self.angle + off, self.LONG_RANGE)
            state.append(max(0.0, min(1.0, d / self.LONG_RANGE)))
        return state  # 23 dims

    def get_lidar_hits(self):
        short_hits = []
        for off in self.SHORT_OFFSETS:
            d, hx, hy = self._cast_ray(self.x, self.y, self.angle + off, self.SHORT_RANGE)
            short_hits.append({'x': round(hx, 1), 'y': round(hy, 1)})
        long_hits = []
        for off in self.LONG_OFFSETS:
            d, hx, hy = self._cast_ray(self.x, self.y, self.angle + off, self.LONG_RANGE)
            long_hits.append({'x': round(hx, 1), 'y': round(hy, 1)})
        return short_hits, long_hits

    def step(self, action, dt):
        # Time accumulates every step
        if not self.finished:
            self.lap_time += dt

        prev_x, prev_y = self.x, self.y

        # Map action to inputs
        accel, steer = 0.0, 0.0
        if action == 0:   accel = 1.0
        elif action == 1: accel = -0.4
        elif action == 2: steer = -1.0
        elif action == 3: steer = 1.0
        elif action == 4: accel, steer = 1.0, -1.0
        elif action == 5: accel, steer = 1.0,  1.0

        # Surface under car
        col = self._pixel(self.x, self.y)
        fm = self._friction_mult(col)

        # Speed update
        self.speed += accel * self.ACCELERATION * dt
        fric = self.FRICTION if accel != 0.0 else self.FRICTION * fm
        if self.speed > 0:
            self.speed = max(0.0, self.speed - fric * dt)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + fric * dt)

        max_s = self.MAX_SPEED * (0.5 if fm > 2.0 else 1.0)
        self.speed = max(-max_s * 0.5, min(max_s, self.speed))

        # Steering
        if abs(self.speed) > 1.0:
            sf = 1.0 / (1.0 + abs(self.speed) / self.MAX_SPEED * self.TURN_FACTOR)
            sign = 1.0 if self.speed > 0 else -1.0
            self.angle += steer * self.TURN_BASE * sf * dt * sign

        # Position update
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

        # Wall collision
        col = self._pixel(self.x, self.y)
        if self._is_wall(col):
            self.x, self.y = prev_x, prev_y
            self.speed *= -0.3
            self.wall_hits += 1

        # Checkpoint crossing (exact C++ logic)
        if not self.finished:
            (cx1, cy1), (cx2, cy2) = self.CHECKPOINTS[self.next_cp]
            if self._seg_cross(prev_x, prev_y, self.x, self.y, cx1, cy1, cx2, cy2):
                if self.next_cp == 0:
                    if self.current_lap > 0:
                        if all(self.cp_crossed[1:]):
                            self.cp_crossed[0] = True
                            self.lap_times.append(self.lap_time)
                            if self.lap_time < self.best_lap:
                                self.best_lap = self.lap_time
                            self.current_lap += 1
                            self.lap_time = 0.0
                            self.cp_crossed = [False] * len(self.CHECKPOINTS)
                            self.next_cp = 1
                            if self.current_lap >= self.TOTAL_LAPS:
                                self.finished = True
                    else:
                        self.current_lap = 1
                        self.lap_time = 0.0
                        self.cp_crossed[0] = False
                        self.next_cp = 1
                else:
                    if self.current_lap > 0:
                        self.cp_crossed[self.next_cp] = True
                        self.next_cp = (self.next_cp + 1) % len(self.CHECKPOINTS)

        self.steps += 1
