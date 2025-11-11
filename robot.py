
from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import time
import os

class Robot():

    def __init__(self):
        

        # -- Debugging --
        self.frame_count = 0
        self.debug_save_dir = "debug_frames" # Folder to save images in

        # -- Constants --
        self.angles = np.deg2rad([90, 210, 330])    # angles of the weels
        self.linearSpeed = 0.8                      # m/s linear speed along (dx,dy)
        self.angularVelocity = 0.0                  # rad/s (spin). Set >0 to rotate CCW
        self.wheelRadius = 1.0                      # wheel radius in meters (or set to your real radius)
        self.radius = 1                             # radius of robot

        # -- Runtime
        self.w = 0
        self.cx = 0
        self.cy = 0
        self.dx = 0
        self.dy = 0

        # camera configuration
        self.camera = self.configure_camera()

    def configure_camera(self):
            picam2 = Picamera2()
            picam2.start_preview(Preview.NULL)
            capture_config = picam2.create_still_configuration()
            picam2.configure(capture_config)
            picam2.start()

            time.sleep(2)
            with picam2.controls as ctrl:
                ctrl.AnalogueGain = 1.0
                ctrl.ExposureTime = 400000
            time.sleep(2)

            return picam2
    
    def preprocess_image(self, gray, th_w, th_h):
        """
        # We coud blur before threshold and clean afterwards
        # blur = cv2.GaussianBlur(roi, (5,5), 0)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
        """

        # TODO ensure shape and format of gray
        h, w = gray.shape[:2]
        roi = gray[int(h*th_h):h, int(w*th_w):int(w-w*th_w)]
        # Correct unpacking order
        _, binv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        return binv, roi

    def middle_vector(self, binary):  # binary: 0/255, white = line
        # 1) Moments (single pass, O(N))
        M = cv2.moments(binary, binaryImage=True)
        if M["m00"] == 0:
            raise ValueError("No white pixels found")

        ## if there is a lot of noise, consider filtering by area first
        # if M["m00"] > 100:
                # TODO Compute intersection movement
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
        dx, dy = float(v[0]), float(v[1])

        return cx, cy, dx, dy

    def draw_middle_vector(self, image, center, direction, scale=100, color=(0,255,0), thickness=2):
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
        # ensure we are drawing on a color image
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

    def draw_debug_info(self, image: np.ndarray, cx: float, cy: float, dx: float, dy: float) -> np.ndarray:
        """
        Draws the centroid and direction vector onto an image for debugging.
        Returns a new BGR image with info drawn on it.
        """
        # We need a color image to draw in color
        if len(image.shape) == 2:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = image.copy() # Make a copy

        cx_int, cy_int = int(cx), int(cy)
        cv2.circle(debug_img, (cx_int, cy_int), 5, (0, 0, 255), -1) # Red centroid

        p2_x = int(cx_int + dx * 50) # Scale the vector by 50px
        p2_y = int(cy_int + dy * 50)
        cv2.line(debug_img, (cx_int, cy_int), (p2_x, p2_y), (255, 0, 0), 2) # Blue line
        
        return debug_img

    def debug_save_images(self, images: dict, save_dir: str, frame_id: int):
        """
        Saves multiple images to a specified directory.
        
        Args:
            images: A dictionary of {"file_suffix": image_array}.
            save_dir: The directory to save images to (e.g., "debug_frames").
            frame_id: The current frame number (e.g., 1, 2, 3...).
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        for suffix, img in images.items():
            if img is not None:
                # Create a unique filename, e.g., "debug_frames/frame_001_Mask.jpg"
                filename = os.path.join(save_dir, f"frame_{frame_id:03d}_{suffix}.jpg")
                cv2.imwrite(filename, img)

    def get_motorW(self):

        # wheel angles around the robot
        x = self.radius * np.cos(self.angles)
        y = self.radius * np.sin(self.angles)

        x1, x2, x3 = x
        y1, y2, y3 = y

        # drive directions = perpendiculars (90 CCW): (nx, ny) = (-y, x)
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


        vx, vy = self.linearSpeed*self.dx, self.linearSpeed*self.dy
        v = np.array([vx, vy, self.angularVelocity], dtype=float)

        # wheel linear speeds (if r=1) or angular speeds (rad/s) if you divide by real r
        w = (M @ v) / self.wheelRadius
        print("wheel speeds:", w)
        return w

    

    def capture_frame_gray(self):
        im = self.camera.capture_array()
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.camera.capture_file("demo.jpg")
        return grey

    def apply_wheel_speeds(self):
        raise NotImplementedError

    def run(self):
        while True:
            # Increment frame counter
            self.frame_count += 1
            
            # --- 1. Get Image ---
            img_gray = self.capture_frame_gray()
            
            # --- 2. Preprocess ---
            mask, roi = self.preprocess_image(img_gray, th_w=0.35, th_h=0.6)
            
            debug_overlay = None # Placeholder
            
            # --- 3. Handle line detection ---
            try:
                cx, cy, dx, dy = self.middle_vector(mask)
                self.cx = cx
                self.cy = cy
                self.dx = dx
                self.dy = dy
                
                # We only need to draw if we found a line
                debug_overlay = self.draw_debug_info(roi, cx, cy, dx, dy)

            except ValueError:
                print("Line lost, continuing with last known direction...")
                pass # debug_overlay remains None

            
            # --- 4. DEBUG SAVING (every 10th frame) ---
            if self.frame_count % 10 == 0:
                print(f"Saving debug frames for frame {self.frame_count}...")
                self.debug_save_images(
                    images={
                        "1_Gray": img_gray,
                        "2_ROI": roi,
                        "3_Mask": mask,
                        "4_Overlay": debug_overlay
                    },
                    save_dir=self.debug_save_dir,
                    frame_id=self.frame_count
                )
            
            # --- 5. Direction ---
            w = self.get_motorW()
            # apply_wheel_speeds(w)

if __name__ == "__main__":  
    robot = Robot()
    robot.run()