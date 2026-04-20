import cv2
import numpy as np
import time

def get_hdr_chessboard(cap, w_expos= 7, b_expos= 9, focus= 0):
    """
    Captures two frames with different exposures and blends them using Mertens fusion.
    """
    w_expos -= 13 
    b_expos -= 13


    # 1. Lock Focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    
    # 2. Disable Auto Exposure
    # Windows: Use 0.25 for manual. Linux: Use 1.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 


    def capture_at_exposure(exp_val):
        cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        # Flush the buffer: Webcams often have 3-5 frames 
        # already queued with old exposure settings.
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        return ret, frame

    # Capture frames for both piece types
    ret_w, img_white = capture_at_exposure(w_expos)
    ret_b, img_black = capture_at_exposure(b_expos)

    if not ret_w or not ret_b:
        print("Error: Could not capture frames.")
        return None

    # 3. Merge using Mertens Exposure Fusion
    # This algorithm is ideal because it doesn't require 
    # camera response function calibration.
    merge_mertens = cv2.createMergeMertens()
    images = [img_white, img_black]
    
    # process() returns a 32-bit float image [0.0, 1.0]
    fusion = merge_mertens.process(images)

    # 4. Convert back to 8-bit [0, 255]
    fusion_8bit = np.clip(fusion * 255, 0, 255).astype('uint8')

    return fusion_8bit

# Example usage:
# result = get_hdr_chessboard(cap, w_expos=-7, b_expos=-5, focus=35)
