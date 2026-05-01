import cv2
import numpy as np
import sys
import os

# Path fix
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
THRESHOLD = 0.75          # kept for compatibility (not used in feature mode)
CAMERA_INDEX = 0

CROP_LEFT_PX = 100
CROP_RIGHT_PX = 80
BOARD_SHRINK_FACTOR = 0.92

# Now represents MIN_GOOD_MATCHES (tune these!)
PIECE_THRESHOLDS = {
    'white_pawn': 6,   'black_pawn': 6,
    'white_rook': 8,   'black_rook': 8,
    'white_knight': 7, 'black_knight': 7,
    'white_bishop': 7, 'black_bishop': 7,
    'white_queen': 9,  'black_queen': 9,
    'white_king': 10,  'black_king': 10,
}

piece_templates = {}
if not os.path.isdir(TEMPLATES_DIR):
    print(f"❌ ERROR: TEMPLATES_DIR does not exist: {TEMPLATES_DIR}")
else:
    for piece_name in os.listdir(TEMPLATES_DIR):
        piece_dir = os.path.join(TEMPLATES_DIR, piece_name)
        if not os.path.isdir(piece_dir):
            continue
        templates_list = []
        for filename in os.listdir(piece_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                template_path = os.path.join(piece_dir, filename)
                template = cv2.imread(template_path, 0)
                if template is not None:
                    templates_list.append(template)
        if templates_list:
            piece_templates[piece_name] = templates_list

print(f"✅ Loaded templates for {len(piece_templates)} piece types")

# ================== FEATURE MATCHING SETUP ==================
orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, nlevels=8)
RATIO_TEST = 0.75
MIN_GLOBAL_MATCHES = 5

template_features = {}
for name, templates in piece_templates.items():
    template_features[name] = []
    for tmpl in templates:
        kp, des = orb.detectAndCompute(tmpl, None)
        if des is not None and len(kp) >= 5:
            template_features[name].append((kp, des))

print(f"✅ Pre-computed ORB features for {len(template_features)} piece types")


def get_board_corners(frame):
    """Unchanged from your original (goodFeaturesToTrack + shrink logic)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    debug_frame = frame.copy()

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.13,
                                      minDistance=40, blockSize=7)
    if corners is None or len(corners) < 20:
        h, w = gray.shape
        cv2.line(debug_frame, (0,0), (w,h), (0,0,255), 15)
        cv2.line(debug_frame, (w,0), (0,h), (0,0,255), 15)
        cv2.imshow("Board Detection FAILED", debug_frame)
        raise RuntimeError("Not enough corners detected.")

    corners = corners.reshape(-1, 2)
    for pt in corners:
        cv2.circle(debug_frame, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    outer_corners = order_points(hull)
    center = outer_corners.mean(axis=0)
    expanded = [center + (pt - center) * 1.12 for pt in outer_corners]
    outer_corners = np.float32(expanded)

    center = outer_corners.mean(axis=0)
    shrunk = [center + (pt - center) * BOARD_SHRINK_FACTOR for pt in outer_corners]
    outer_corners = np.float32(shrunk)

    cv2.polylines(debug_frame, [outer_corners.astype(int).reshape((-1,1,2))],
                  isClosed=True, color=(0, 255, 0), thickness=4)
    print(f"✅ Detected {len(corners)} corners")
    return outer_corners, debug_frame


def rectify_board(frame, src_points, output_size=800):
    """Unchanged."""
    src_points = np.float32(src_points).reshape(-1, 2)
    side = int(output_size)
    dst_points = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    H, _ = cv2.findHomography(src_points, dst_points)
    rectified = cv2.warpPerspective(frame, H, (side, side))
    return rectified, H, side


def detect_pieces(rectified, square_size_px):
    """Feature matching version (replaces template matching)."""
    gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for r in range(8):
        for c in range(8):
            margin = int(square_size_px * 0.08)
            x1 = max(0, int(c * square_size_px) - margin)
            y1 = max(0, int(r * square_size_px) - margin)
            x2 = min(gray_rect.shape[1], int((c + 1) * square_size_px) + margin)
            y2 = min(gray_rect.shape[0], int((r + 1) * square_size_px) + margin)
            square = gray_rect[y1:y2, x1:x2]

            if square.size < 100:
                continue

            kp, des = orb.detectAndCompute(square, None)
            if des is None or len(kp) < 6:
                continue

            best_score = 0
            best_name = '.'

            for name, feats in template_features.items():
                min_matches = PIECE_THRESHOLDS.get(name, MIN_GLOBAL_MATCHES)
                for kp_t, des_t in feats:
                    if des_t is None or len(des_t) < 5:
                        continue
                    matches = bf.knnMatch(des, des_t, k=2)
                    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
                    score = len(good)
                    if score > best_score and score >= min_matches:
                        best_score = score
                        best_name = name

            if best_name != '.':
                board_state[r][c] = best_name

    return board_state


def draw_piece_labels(rectified, board_state, square_size_px):
    """Unchanged from your original."""
    labeled = rectified.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_color = (0, 255, 255)

    short_names = {
        'white_pawn': 'wP', 'black_pawn': 'bP',
        'white_rook': 'wR', 'black_rook': 'bR',
        'white_knight': 'wN', 'black_knight': 'bN',
        'white_bishop': 'wB', 'black_bishop': 'bB',
        'white_queen': 'wQ', 'black_queen': 'bQ',
        'white_king': 'wK', 'black_king': 'bK',
    }

    for r in range(8):
        for c in range(8):
            name = board_state[r][c]
            if name != '.':
                cx = int((c + 0.5) * square_size_px)
                cy = int((r + 0.5) * square_size_px)
                text_y = max(15, cy - int(square_size_px * 0.38))
                short = short_names.get(name, name[:2].upper())
                (tw, th), _ = cv2.getTextSize(short, font, font_scale, thickness)
                text_x = cx - tw // 2
                cv2.putText(labeled, short, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(labeled, short, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return labeled


def get_feature_match_debug_image(rectified, board_state, square_size_px, max_pieces=6):
    """Creates a nice debug visualization showing actual feature matches."""
    gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    detected = [(r, c, board_state[r][c]) for r in range(8) for c in range(8) if board_state[r][c] != '.']
    if not detected:
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        cv2.putText(img, "No pieces detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    detected = detected[:max_pieces]
    panel_h, panel_w = 300, 600
    canvas = np.zeros((panel_h * 2, panel_w * 3, 3), dtype=np.uint8)

    for idx, (r, c, name) in enumerate(detected):
        row = idx // 3
        col = idx % 3
        y_off = row * panel_h
        x_off = col * panel_w

        margin = int(square_size_px * 0.08)
        x1 = max(0, int(c * square_size_px) - margin)
        y1 = max(0, int(r * square_size_px) - margin)
        x2 = min(gray_rect.shape[1], int((c + 1) * square_size_px) + margin)
        y2 = min(gray_rect.shape[0], int((r + 1) * square_size_px) + margin)
        square = gray_rect[y1:y2, x1:x2]
        square_color = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)

        # Find best template again for visualization
        kp_sq, des_sq = orb.detectAndCompute(square, None)
        best_score = 0
        best_kp_t = best_des_t = best_tmpl = None

        for tmpl_name, feats in template_features.items():
            if tmpl_name != name:
                continue
            for kp_t, des_t in feats:
                if des_t is None:
                    continue
                matches = bf.knnMatch(des_sq, des_t, k=2)
                good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
                if len(good) > best_score:
                    best_score = len(good)
                    best_kp_t, best_des_t, best_tmpl = kp_t, des_t, tmpl

        if best_tmpl is None:
            continue

        # Draw matches
        match_img = cv2.drawMatches(
            square_color, kp_sq,
            cv2.cvtColor(best_tmpl, cv2.COLOR_GRAY2BGR), best_kp_t,
            [m for m, n in bf.knnMatch(des_sq, best_des_t, k=2)
             if m.distance < RATIO_TEST * n.distance][:30],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Resize to panel
        match_img = cv2.resize(match_img, (panel_w, panel_h))
        canvas[y_off:y_off+panel_h, x_off:x_off+panel_w] = match_img

        # Label
        cv2.putText(canvas, f"{name} ({best_score} matches) @ ({r},{c})",
                    (x_off + 10, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return canvas


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)

    print("=== board_vision.py — Feature Matching + Debug Mode ===")
    print("Windows: Live Original | HDR Fused | Debug Corners | Detected Pieces | Feature Matches Debug")
    print("Press 'q' to quit.")

    cv2.namedWindow("Live Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HDR Fused", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug Corners", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detected Pieces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Feature Matches Debug", cv2.WINDOW_NORMAL)

    while True:
        if get_hdr_chessboard is not None:
            hdr_frame = get_hdr_chessboard(cap, w_expos=-5, b_expos=-5, focus=35)
            if hdr_frame is None:
                ret, frame = cap.read()
                hdr_frame = frame
        else:
            ret, hdr_frame = cap.read()
            frame = hdr_frame

        if hdr_frame is None:
            break

        h, w = hdr_frame.shape[:2]
        hdr_frame = hdr_frame[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()
        if 'frame' in locals() and frame is not None:
            frame = frame[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()

        try:
            src_points, debug_frame = get_board_corners(hdr_frame)
            rectified, H, side_px = rectify_board(hdr_frame, src_points)
            square_size_px = side_px / 8.0

            board_state = detect_pieces(rectified, square_size_px)
            labeled_board = draw_piece_labels(rectified, board_state, square_size_px)

            # NEW: Feature matches debug
            match_debug = get_feature_match_debug_image(rectified, board_state, square_size_px)
            cv2.imshow("Feature Matches Debug", match_debug)

            print("\nBoard state:")
            for row in board_state:
                print(" ".join(f"{piece:12}" for piece in row))

            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)
            cv2.imshow("Debug Corners", debug_frame)
            cv2.imshow("Detected Pieces", labeled_board)

        except RuntimeError as e:
            print(f"❌ {e}")
            cv2.imshow("Live Original", frame if 'frame' in locals() else hdr_frame)
            cv2.imshow("HDR Fused", hdr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()