import cv2
import numpy as np
import sys
import os

# Path fix similar to vision_move_detector.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import HDR function
try:
    from make_hdr import get_hdr_chessboard
    print("✅ Successfully imported HDR fusion from make_hdr.py")
except Exception as e:
    print(f"❌ Failed to import make_hdr: {e}")
    get_hdr_chessboard = None

# ================== CONFIG ==================
BOARD_SIZE_CM = 40.0
SQUARE_SIZE_CM = BOARD_SIZE_CM / 8.0
TEMPLATES_DIR = r"C:\Users\tkubi\Documents\GithubRepos\CAD Applications Final Project\6DOF_Chess_Bot\ChessStuff\piece_templates"
THRESHOLD = 0.75
CAMERA_INDEX = 0

# === Editable crop values (left & right margins) ===
CROP_LEFT_PX = 100
CROP_RIGHT_PX = 80

# Per-piece thresholds (tune these based on your templates and lighting)
PIECE_THRESHOLDS = {
    'white_pawn': 0.78, 'black_pawn': 0.78,
    'white_rook': 0.75, 'black_rook': 0.75,
    'white_knight': 0.72, 'black_knight': 0.72,
    'white_bishop': 0.74, 'black_bishop': 0.74,
    'white_queen': 0.70, 'black_queen': 0.70,
    'white_king': 0.68, 'black_king': 0.68,
}

# Piece templates
piece_templates = {}
for filename in os.listdir(TEMPLATES_DIR):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        name = os.path.splitext(filename)[0]
        template_path = os.path.join(TEMPLATES_DIR, filename)
        template = cv2.imread(template_path, 0)  # load as grayscale
        if template is not None:
            piece_templates[name] = template
        else:
            print(f"Warning: Could not load template {filename}")

print(f"✅ Loaded {len(piece_templates)} piece templates: {list(piece_templates.keys())}")
if not piece_templates:
    print("⚠️  WARNING: No templates were loaded from", TEMPLATES_DIR)

def get_board_corners(frame):
    """Detect board corners using cv2.goodFeaturesToTrack instead of findChessboardCorners.
    This is experimental and may not be as reliable as chessboard-specific detection."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # === CLAHE contrast enhancement ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    debug_frame = frame.copy()
    
    # === Use goodFeaturesToTrack to find corners ===
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=120,
        qualityLevel=0.13,
        minDistance=40,
        blockSize=7
    )
    
    if corners is None or len(corners) < 20:
        print("ERROR: Not enough corners detected with goodFeaturesToTrack!")
        h, w = gray.shape
        cv2.line(debug_frame, (0,0), (w,h), (0,0,255), 15)
        cv2.line(debug_frame, (w,0), (0,h), (0,0,255), 15)
        cv2.imshow("Board Detection FAILED", debug_frame)
        raise RuntimeError("Not enough corners detected. Improve lighting / camera alignment.")
    
    corners = corners.reshape(-1, 2)
    
    # Draw all detected corners in red
    for pt in corners:
        cv2.circle(debug_frame, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
    
    # === Robust outer corner calculation (clean 8x8 box) ===
    # Use convex hull + proper corner ordering (much more reliable)
    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect

    outer_corners = order_points(hull)

    # Slightly expand the box so we capture the full outer squares (not just inner corners)
    center = outer_corners.mean(axis=0)
    expanded = []
    for pt in outer_corners:
        vec = pt - center
        expanded.append(center + vec * 1.10)   # 12% expansion — tweak if needed
    outer_corners = np.float32(expanded)
    
    # Draw green box around estimated board area
    cv2.polylines(debug_frame, [outer_corners.astype(int).reshape((-1,1,2))], 
                  isClosed=True, color=(0, 255, 0), thickness=4)
    
    print(f"✅ Detected {len(corners)} corners using goodFeaturesToTrack")
    
    return outer_corners, debug_frame


def rectify_board(frame, src_points):
    """Rectify board with strict error handling."""
    if src_points is None or len(src_points) != 4:
        raise ValueError(f"Invalid src_points received: {src_points}")
    
    src_points = np.float32(src_points).reshape(-1, 2)
    side = 800
    dst_points = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    
    H, mask = cv2.findHomography(src_points, dst_points)
    if H is None:
        raise RuntimeError("findHomography failed - points may be collinear or invalid.")
    
    rectified = cv2.warpPerspective(frame, H, (side, side))
    return rectified, H, side


def detect_pieces(rectified, square_size_px):
    """Improved template matching"""
    gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    detections = []

    for name, template in piece_templates.items():
        if template is None:
            continue
        thresh = PIECE_THRESHOLDS.get(name, THRESHOLD)
        res = cv2.matchTemplate(gray_rect, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh)
        for pt in zip(*loc[::-1]):
            col = int(pt[0] // square_size_px)
            row = int(pt[1] // square_size_px)
            if 0 <= row < 8 and 0 <= col < 8:
                score = float(res[pt[1], pt[0]])
                detections.append((score, row, col, name))

    best_per_square = {}
    for score, row, col, name in detections:
        key = (row, col)
        if key not in best_per_square or score > best_per_square[key][0]:
            best_per_square[key] = (score, name)

    for (row, col), (score, name) in best_per_square.items():
        board_state[row][col] = name

    return board_state


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)

    print("=== board_vision2.py - Using goodFeaturesToTrack (Experimental) ===")
    print("Windows: Live Original | HDR Fused | Debug Corners")
    print("Press 'q' to quit.")

    cv2.namedWindow("Live Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HDR Fused", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug Corners", cv2.WINDOW_NORMAL)

    while True:
        # Capture HDR frame
        if get_hdr_chessboard is not None:
            hdr_frame = get_hdr_chessboard(cap, w_expos=-5, b_expos=-5, focus=35)
            if hdr_frame is None:
                ret, frame = cap.read()
                hdr_frame = frame
        else:
            ret, hdr_frame = cap.read()
            frame = hdr_frame

        if hdr_frame is None:
            print("Failed to capture frame.")
            break

        # === Crop left and right sides to eliminate fringe false positives ===
        h, w = hdr_frame.shape[:2]
        hdr_frame = hdr_frame[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()
        if 'frame' in locals() and frame is not None:
            frame = frame[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()

        try:
            src_points, debug_frame = get_board_corners(hdr_frame)

            rectified, H, side_px = rectify_board(hdr_frame, src_points)
            square_size_px = side_px / 8.0

            board_state = detect_pieces(rectified, square_size_px)

            print("\nBoard state:")
            for row in board_state:
                print(" ".join(f"{piece:12}" for piece in row))

            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)
            cv2.imshow("Debug Corners", debug_frame)

        except RuntimeError as e:
            print(f"❌ {e}")
            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()