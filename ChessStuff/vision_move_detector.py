import sys
import os

# ================== ROBUST PATH FIX ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print("=== Vision Move Detector Startup Diagnostics ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {current_dir}")
print(f"Project root added to sys.path: {project_root}")

# ================== HDR + Cropping Support ==================
CROP_LEFT_PX = 100
CROP_RIGHT_PX = 80

try:
    from .make_hdr import get_hdr_chessboard
    print("✅ Successfully imported HDR fusion from make_hdr.py")
except Exception as e:
    print(f"❌ Failed to import make_hdr: {e}")
    get_hdr_chessboard = None

try:
    from ChessStuff.board_vision import (
        get_board_corners,
        rectify_board,
        detect_pieces,
        draw_piece_labels,
        THRESHOLD,
        CAMERA_INDEX
    )
    print("✅ Successfully imported from ChessStuff.board_vision")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

import cv2
import numpy as np

PIECE_MAP = {
    'white_pawn': 'P', 'white_rook': 'R', 'white_knight': 'N',
    'white_bishop': 'B', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_rook': 'r', 'black_knight': 'n',
    'black_bishop': 'b', 'black_queen': 'q', 'black_king': 'k',
}


def get_good_frame(cap):
    """Capture HDR-fused + cropped frame (same as board_vision standalone)."""
    if get_hdr_chessboard is not None:
        frame = get_hdr_chessboard(cap, w_expos=-5, b_expos=-5, focus=35)
        if frame is None:
            ret, frame = cap.read()
    else:
        ret, frame = cap.read()

    if frame is None:
        return None

    h, w = frame.shape[:2]
    frame = frame[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()
    return frame


def get_8x8_board_from_frame(frame):
    """
    Returns:
        board_fen, rectified, debug_frame
    """
    src_points, debug_frame = get_board_corners(frame)   # ← now gets debug frame too
    rectified, H, side_px = rectify_board(frame, src_points)
    square_size_px = side_px / 8.0

    board_state_names = detect_pieces(rectified, square_size_px)

    # Draw and display labeled board (small text above each piece) for visual feedback.
    # This appears in BOTH standalone board_vision.py AND the main.py / vision_move_detector pipeline.
    labeled_board = draw_piece_labels(rectified, board_state_names, square_size_px)
    cv2.namedWindow("Detected Pieces", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Pieces", labeled_board)
    cv2.waitKey(1)

    board_fen = [['.' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            name = board_state_names[r][c]
            board_fen[r][c] = PIECE_MAP.get(name, '.')
    return board_fen, rectified, debug_frame


def detect_move_uci(board_before, board_after):
    from_sq = to_sq = None
    files = 'abcdefgh'
    ranks = '87654321'

    for r in range(8):
        for c in range(8):
            if board_before[r][c] != board_after[r][c]:
                if board_before[r][c] != '.' and board_after[r][c] == '.':
                    from_sq = (r, c)
                elif board_before[r][c] == '.' and board_after[r][c] != '.':
                    to_sq = (r, c)
                elif board_before[r][c] != '.' and board_after[r][c] != '.':
                    from_sq = (r, c)
                    to_sq = (r, c)

    if from_sq and to_sq:
        from_uci = files[from_sq[1]] + ranks[from_sq[0]]
        to_uci = files[to_sq[1]] + ranks[to_sq[0]]
        return from_uci + to_uci
    return None


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\n=== Chess Vision Move Detector ===")
    print("Press 'b' / 'a' / 'q'\n")

    board_before = None
    before_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('b') and not before_captured:
            good_frame = get_good_frame(cap)
            if good_frame is None:
                continue
            board_before, rectified_before, debug_before = get_8x8_board_from_frame(good_frame)
            cv2.imwrite("before.jpg", good_frame)
            cv2.imshow("Debug Corners", debug_before)   # show debug
            print("✅ BEFORE captured")
            before_captured = True

        elif key == ord('a') and before_captured:
            good_frame = get_good_frame(cap)
            if good_frame is None:
                continue
            board_after, rectified_after, debug_after = get_8x8_board_from_frame(good_frame)
            cv2.imwrite("after.jpg", good_frame)
            cv2.imshow("Debug Corners", debug_after)

            move_uci = detect_move_uci(board_before, board_after)
            print("\n=== DETECTED MOVE ===")
            print(move_uci or "No move detected")
            before_captured = False
            board_before = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()