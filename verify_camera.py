"""
Camera accuracy verification: detect a green object and report its X, Y, Z
position in the CAMERA OPTICAL FRAME (meters), using RealSense D415 RGB + depth.

Convention (camera optical frame):
  X = right, Y = down, Z = forward (distance from camera), origin at camera lens.

How to verify accuracy:
  1. Place a green object at a KNOWN measured distance (e.g. exactly 30cm from
     camera, measured with a ruler/tape along the camera's forward axis).
  2. Run this script, read the printed/on-screen Z value.
  3. Compare: if ruler says 0.300m and script says 0.295-0.305m, that's normal
     (D415 spec accuracy is roughly +/-1-2% of distance, worse up close <30cm
     and beyond 3m). Bigger errors mean something is off (see notes at bottom).

Press 'q' to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------- CONFIG ----------------
WIDTH, HEIGHT, FPS = 640, 480, 30   # lower res = more stable depth, less noise
MIN_CONTOUR_AREA = 300              # ignore tiny green noise blobs (pixels^2)

# HSV green range - adjust if your object isn't detected well.
# Tip: bright/matte green works far better than glossy green (glare breaks detection).
GREEN_LOWER = np.array([40, 70, 70])
GREEN_UPPER = np.array([85, 255, 255])

# ---------------- INIT REALSENSE ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)

# Align depth to color so pixel (u,v) means the same physical point in both
align = rs.align(rs.stream.color)

# Get depth scale (raw depth units -> meters)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
print(f"Camera intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f}")
print(f"Depth scale: {depth_scale} meters/unit")

# Optional: a simple temporal filter to reduce depth noise/flicker in the readout
depth_history = []
HISTORY_LEN = 5


def get_median_depth(depth_frame, u, v, window=3):
    """Median depth over a small window around (u,v) in raw units, robust to single-pixel dropouts."""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    vals = []
    for dv in range(-window, window + 1):
        for du in range(-window, window + 1):
            uu, vv = u + du, v + dv
            if 0 <= uu < w and 0 <= vv < h:
                d = depth_frame.get_distance(uu, vv)  # already in meters
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = color_img.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area >= MIN_CONTOUR_AREA:
                (x_px, y_px), radius = cv2.minEnclosingCircle(largest)
                u, v = int(x_px), int(y_px)

                Z = get_median_depth(depth_frame, u, v, window=3)

                if Z > 0:
                    # Deproject pixel + depth -> 3D point in camera optical frame
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], Z)

                    cv2.circle(display, (u, v), int(radius), (0, 255, 0), 2)
                    cv2.circle(display, (u, v), 4, (0, 0, 255), -1)

                    text_lines = [
                        f"X: {X*100:6.2f} cm  (right+)",
                        f"Y: {Y*100:6.2f} cm  (down+)",
                        f"Z: {Z*100:6.2f} cm  (distance)",
                    ]
                    for i, line in enumerate(text_lines):
                        cv2.putText(display, line, (10, 30 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    print(f"X={X*100:6.2f}cm  Y={Y*100:6.2f}cm  Z={Z*100:6.2f}cm   "
                          f"(pixel u={u} v={v}, contour_area={int(area)}px)")
                else:
                    cv2.putText(display, "No valid depth at this pixel", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Green object too small/not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No green object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Color - green object XYZ", display)
        cv2.imshow("Green mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
