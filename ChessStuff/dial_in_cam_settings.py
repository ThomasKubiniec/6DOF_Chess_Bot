import cv2

cap = cv2.VideoCapture(0)

# Disable automatic settings first
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Use 0.25 if on Windows


def update(val): pass

# Create a window with trackbars to find your "sweet spot"
cv2.namedWindow("Tuner")
cv2.createTrackbar("Focus", "Tuner", 0, 255, update) # 30, 255
cv2.createTrackbar("Exposure", "Tuner", 0, 13, update) # 5, 13 # Scale 0-13 mapping to negative values

while True:
    f = cv2.getTrackbarPos("Focus", "Tuner")
    e = cv2.getTrackbarPos("Exposure", "Tuner")
    
    cap.set(cv2.CAP_PROP_FOCUS, f)
    # Map trackbar 0-13 to exposure -13 to 0
    cap.set(cv2.CAP_PROP_EXPOSURE, (e - 13)) 
    
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("Chessboard View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Final Settings - Focus: {f}, Exposure: {e-13}")
        break

cap.release()
cv2.destroyAllWindows()
