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
    """Detect chessboard corners using the more robust SB detector.
    Raises clear error if detection fails (no fallback to garbage data)."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern_size = (7, 7)
    
    debug_frame = frame.copy()
    
    # === PRIMARY: Try the modern robust detector first ===
    # Safe flags only - NORMALIZE_IMAGE is not accepted by findChessboardCornersSB in some OpenCV versions
    ret = False
    corners = None
    try:
        ret, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_EXHAUSTIVE
        )
    except cv2.error:
        ret = False
        corners = None
    
    # === FALLBACK: Old detector if SB fails or returns invalid result ===
    if not ret or corners is None or len(corners) < 49:
        print("SB detector failed — trying classic findChessboardCorners...")
        flags_list = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]
        for flags in flags_list:
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
            if ret and len(corners) >= 49:
                break
    
    # Still nothing? Try inverted image with SB
    if not ret or corners is None or len(corners) < 49:
        print("Trying inverted image with SB detector...")
        gray_inv = 255 - gray
        try:
            ret, corners = cv2.findChessboardCornersSB(
                gray_inv,
                pattern_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_EXHAUSTIVE
            )
        except cv2.error:
            ret = False
            corners = None
    
    if not ret or corners is None or len(corners) < 49:
        print("ERROR: Chessboard corners not detected!")
        h, w = gray.shape
        cv2.line(debug_frame, (0,0), (w,h), (0,0,255), 15)
        cv2.line(debug_frame, (w,0), (0,h), (0,0,255), 15)
        cv2.imshow("Chessboard Detection FAILED", debug_frame)
        raise RuntimeError("Chessboard corners not detected. Improve lighting / camera alignment.")
    
    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    corners = corners.reshape(-1, 2)
    
    # === CRITICAL SANITY CHECK - Reject tiny/noisy detections ===
    horiz_diffs = np.diff(corners[::7, 0])
    vert_diffs = np.diff(corners[:7, 1])
    sq_size_x = np.mean(horiz_diffs) if len(horiz_diffs) > 0 else 0
    sq_size_y = np.mean(vert_diffs) if len(vert_diffs) > 0 else 0
    
    if sq_size_x < 15 or sq_size_y < 15:
        print(f"❌ REJECTED: Detected square size too small ({sq_size_x:.1f}, {sq_size_y:.1f} pixels)")
        print("   This is almost certainly noise or a reflection, NOT your chessboard.")
        h, w = gray.shape
        cv2.line(debug_frame, (0,0), (w,h), (0,0,255), 15)
        cv2.line(debug_frame, (w,0), (0,h), (0,0,255), 15)
        cv2.imshow("Chessboard Detection FAILED (too small)", debug_frame)
        raise RuntimeError(f"Detected grid too small ({sq_size_x:.1f}x{sq_size_y:.1f}px). Move camera closer or improve contrast/lighting.")
    
    # === VISUALIZATION ===
    for pt in corners:
        cv2.circle(debug_frame, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
    
    for i in range(7):
        cv2.line(debug_frame, tuple(corners[i*7].astype(int)), tuple(corners[i*7+6].astype(int)), (0,0,255), 2)
        cv2.line(debug_frame, tuple(corners[i].astype(int)), tuple(corners[42+i].astype(int)), (0,0,255), 2)

    # === CORRECT OUTER CORNER CALCULATION FOR FULL 8x8 BOARD ===
    tl_inner = corners[0]
    tr_inner = corners[6]
    br_inner = corners[-1]
    bl_inner = corners[42]

    top_dir = tr_inner - tl_inner
    left_dir = bl_inner - tl_inner
    bottom_dir = br_inner - bl_inner
    right_dir = br_inner - tr_inner

    half_sq_x = sq_size_x * 0.5
    half_sq_y = sq_size_y * 0.5
    
    top_left_outer = tl_inner - (top_dir / np.linalg.norm(top_dir) * half_sq_x * 0.6 + 
                                  left_dir / np.linalg.norm(left_dir) * half_sq_y * 0.6)
    top_right_outer = tr_inner + (top_dir / np.linalg.norm(top_dir) * half_sq_x * 0.6 - 
                                   right_dir / np.linalg.norm(right_dir) * half_sq_y * 0.6)
    bottom_right_outer = br_inner + (bottom_dir / np.linalg.norm(bottom_dir) * half_sq_x * 0.6 + 
                                      right_dir / np.linalg.norm(right_dir) * half_sq_y * 0.6)
    bottom_left_outer = bl_inner - (bottom_dir / np.linalg.norm(bottom_dir) * half_sq_x * 0.6 - 
                                     left_dir / np.linalg.norm(left_dir) * half_sq_y * 0.6)

    outer_corners = np.float32([top_left_outer, top_right_outer, bottom_right_outer, bottom_left_outer])

    # Draw thick green box around FULL 8x8 board
    cv2.polylines(debug_frame, [outer_corners.astype(int).reshape((-1,1,2))], 
                  isClosed=True, color=(0, 255, 0), thickness=5)

    print(f"✅ VALID corners detected (SB) | sq_size ≈ ({sq_size_x:.1f}, {sq_size_y:.1f})")

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
    """Improved template matching:
    - Uses per-piece thresholds
    - Collects all detections
    - Applies per-square non-maximum suppression (keeps best match only)
    - Empty squares remain '.' 
    """
    gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    detections = []  # (score, row, col, name)

    for name, template in piece_templates.items():
        if template is None:
            continue
            
        thresh = PIECE_THRESHOLDS.get(name, THRESHOLD)
        
        res = cv2.matchTemplate(gray_rect, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh)
        
        for pt in zip(*loc[::-1]):  # pt = (x, y) top-left
            col = int(pt[0] // square_size_px)
            row = int(pt[1] // square_size_px)
            if 0 <= row < 8 and 0 <= col < 8:
                score = float(res[pt[1], pt[0]])
                detections.append((score, row, col, name))

    # Non-maximum suppression: keep only the highest-scoring detection per square
    best_per_square = {}
    for score, row, col, name in detections:
        key = (row, col)
        if key not in best_per_square or score > best_per_square[key][0]:
            best_per_square[key] = (score, name)

    # Fill board state with best detections
    for (row, col), (score, name) in best_per_square.items():
        board_state[row][col] = name

    # Optional debug print (uncomment when tuning)
    # print("Piece detections (score > threshold):")
    # for (r, c), (s, n) in sorted(best_per_square.items()):
    #     print(f"  {n:12} at square ({r},{c})  score={s:.3f}")

    return board_state


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)

    print("=== board_vision.py - Live HDR Camera Debug Mode ===")
    print("Using HDR fusion for better lighting.")
    print("Windows: Live Original | HDR Fused | Debug Corners | Rectified")
    print("Press 'q' to quit.")

    cv2.namedWindow("Live Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HDR Fused", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug Corners", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Rectified Board", cv2.WINDOW_NORMAL)

    while True:
        # Capture HDR frame
        if get_hdr_chessboard is not None:
            hdr_frame = get_hdr_chessboard(cap, w_expos=-5, b_expos=-3, focus=35)  # Tune these values!
            if hdr_frame is None:
                ret, frame = cap.read()
                hdr_frame = frame
        else:
            ret, hdr_frame = cap.read()
            frame = hdr_frame

        if hdr_frame is None:
            print("Failed to capture frame.")
            break

        try:
            src_points, debug_frame = get_board_corners(hdr_frame)
            print("✅ Chessboard corners detected successfully with HDR.")

            rectified, H, side_px = rectify_board(hdr_frame, src_points)
            square_size_px = side_px / 8.0

            board_state = detect_pieces(rectified, square_size_px)

            print("\nBoard state:")
            for row in board_state:
                print(" ".join(f"{piece:12}" for piece in row))

            # Show all debug windows
            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)
            cv2.imshow("Debug Corners", debug_frame)
            cv2.imshow("Rectified Board", rectified)

        except RuntimeError as e:
            print(f"❌ {e}")
            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()