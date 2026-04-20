import cv2
import numpy as np

from make_hdr import get_hdr_chessboard

# 0 is default camera, in my case that was my laptop's webcam
# 1 is then the next camera, being the USB periferal webcam
cam = cv2.VideoCapture(0) 

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    

    frame = get_hdr_chessboard(cap= cam, 
                               w_expos= 7,
                               b_expos= 9,
                               focus= 0)


    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()