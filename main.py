from helpers import *  
from picamera2 import Picamera2, Preview

if __name__ == "__main__":  

    # CONSTANTS
    angles = np.deg2rad([90, 210, 330])
    linearSpeed = 0.8          # m/s linear speed along (dx,dy)
    angularVelocity = 0.0      # rad/s (spin). Set >0 to rotate CCW
    wheelRadius = 1.0          # wheel radius in meters (or set to your real radius)
    radius = 1  # radius of robot
    
    # --- Camera ---
    picam2 = Picamera2()
    picam2.start_preview(Preview.NULL)
    capture_config = picam2.create_still_configuration()
    picam2.configure(capture_config)
    picam2.start()

    # this
    time.sleep(2)
    with picam2.controls as ctrl:
        ctrl.AnalogueGain = 1.0
        ctrl.ExposureTime = 400000
    time.sleep(2)

    img_gray = None


    while True:

        # 
        img_gray = capture_frame_gray()
        
        ## preprocess
        mask, roi = preprocess(img_gray, th_w=0.35, th_h=0.6)
        
        # intersections
        # objject
        # line missing 

        cx, cy, dx, dy = middle_vector(mask)
        
        ## direction
        w = get_W(angles, radius, dx, dy, linearSpeed, angularVelocity, wheelRadius)
        apply_wheel_speeds(w)