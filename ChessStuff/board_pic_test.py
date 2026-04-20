import cv2

from make_hdr import get_hdr_chessboard

# Connect to the default camera
# may need to swap between 0 and 1, sometimes they flipflop
cam = cv2.VideoCapture(0)

# Capture a single frame
ret, frame = cam.read()

frame = get_hdr_chessboard(cap= cam,
                           w_expos= 5, # 7
                           b_expos= 8, # 9
                           focus= 0)

if ret:
    # Save the frame to a file
    cv2.imwrite("hdr_loaded_with_lighting.png", frame)
    print("Photo saved successfully!")
else:
    print("Failed to capture image.")

# Release the camera and clean up
cam.release()
cv2.destroyAllWindows()
