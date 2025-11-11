import cv2
import numpy as np
import time
from PIL import Image


def preprocess(gray, th_w, th_h):
    """
    # We coud blur before threshold and clean afterwards
    # blur = cv2.GaussianBlur(roi, (5,5), 0)
    # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
    """
    h, w = gray.shape[:2]
    roi = gray[int(h*th_h):h, int(w*th_w):int(w-w*th_w)]
    # Correct unpacking order
    _, binv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    return binv, roi

def middle_vector(binary):  # binary: 0/255, white = line
    # 1) Moments (single pass, O(N))
    M = cv2.moments(binary, binaryImage=True)
    if M["m00"] == 0:
        raise ValueError("No white pixels found")

    ## if there is a lot of noise, consider filtering by area first
    if M["m00"] > 100:
        ROI = Left side
    # m10 = suma de las coordinadas horizontales
    # m01 = suma de las coordinadas verticales
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 2) Covariance of white pixel distribution (from central moments)
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    cov = np.array([[mu20, mu11],
                    [mu11, mu02]])

    # 3) Principal axis via eigen decomposition (largest eigenvalue)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]      # shape (2,), unit vector in image coords (x right, y down)
    vx, vy = float(v[0]), float(v[1])

    return cx, cy, vx, vy


def draw_middle_vector(image, center, direction, scale=100, color=(0,255,0), thickness=2):
    """
    Draw the middle vector (principal axis) on an image.

    Parameters:
        image: np.ndarray (grayscale or BGR)
        center: (cx, cy) tuple from middle_vector()
        direction: (vx, vy) tuple, unit vector along main axis
        scale: length of the arrow to draw (in pixels)
        color: arrow color (BGR)
        thickness: line thickness

    Returns:
        A copy of the image with the vector drawn on it.
    """
    # ensure we’re drawing on a color image
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    cx, cy = center
    vx, vy = direction

    # arrow endpoints (two directions from the center)
    pt1 = (int(cx - vx * scale), int(cy - vy * scale))
    pt2 = (int(cx + vx * scale), int(cy + vy * scale))

    # draw the main line (green)
    cv2.arrowedLine(vis, pt2, pt1, color, thickness, tipLength=0.2)

    # mark the centroid (red dot)
    cv2.circle(vis, (int(cx), int(cy)), 4, (0,0,255), -1)

    return vis


def get_W(angles, radius, dx, dy, linearSpeed, angularVelocity, wheelRadius):

  # wheel angles around the robot
  x = radius * np.cos(angles)
  y = radius * np.sin(angles)

  x1, x2, x3 = x
  y1, y2, y3 = y

  # drive directions = perpendiculars (90° CCW): (nx, ny) = (-y, x)
  nx1, ny1 = -y1, x1
  nx2, ny2 = -y2, x2
  nx3, ny3 = -y3, x3

  # kinematic matrix (linear rim speed). For this symmetric layout,
  # the rotation coupling term (ny*x - nx*y) = 1 for all three.
  M = np.array([
      [nx1, ny1, ny1*x1 - nx1*y1],
      [nx2, ny2, ny2*x2 - nx2*y2],
      [nx3, ny3, ny3*x3 - nx3*y3],
  ], dtype=float)


  vx, vy = linearSpeed*dx, linearSpeed*dy
  v = np.array([vx, vy, angularVelocity], dtype=float)

  # wheel linear speeds (if r=1) or angular speeds (rad/s) if you divide by real r
  w = (M @ v) / wheelRadius
  print("wheel speeds:", w)
  return w


def capture_frame_gray(picam2):
    return
    picam2.capture_array()

def apply_wheel_speeds(w):
    raise NotImplementedError