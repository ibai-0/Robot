from helpers import *  

if __name__ == "__main__":  

    # CONSTANTS
    angles = np.deg2rad([90, 210, 330])
    linearSpeed = 0.8          # m/s linear speed along (dx,dy)
    angularVelocity = 0.0      # rad/s (spin). Set >0 to rotate CCW
    wheelRadius = 1.0          # wheel radius in meters (or set to your real radius)
    radius = 1  # radius of robot
    
    img_gray = None
    while True:
        img_gray = capture_frame_gray()
        mask, roi = preprocess(img_gray, th_w=0.35, th_h=0.6)
        cx, cy, dx, dy = middle_vector(mask)
        w = get_W(angles, radius, dx, dy, linearSpeed, angularVelocity, wheelRadius)
        apply_wheel_speeds(w)