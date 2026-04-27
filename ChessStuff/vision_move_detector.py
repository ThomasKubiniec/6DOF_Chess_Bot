import sys
import os

# ================== ROBUST PATH FIX ==================
# Ensure we can import sibling modules regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to parent of ChessStuff/
sys.path.insert(0, project_root)  # Add project root so "from ChessStuff.xxx" works

print("=== Vision Move Detector Startup Diagnostics ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {current_dir}")
print(f"Project root added to sys.path: {project_root}")

# Now import (this should now work when running main.py from root)
try:
    from ChessStuff.board_vision import (
        get_board_corners,
        rectify_board,
        detect_pieces,
        THRESHOLD,
        CAMERA_INDEX
    )
    print("✅ Successfully imported from ChessStuff.board_vision")
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("Make sure board_vision.py exists in the ChessStuff/ folder.")
    sys.exit(1)

import cv2
import numpy as np

# Piece name to standard FEN mapping
PIECE_MAP = {
    'white_pawn': 'P', 'white_rook': 'R', 'white_knight': 'N',
    'white_bishop': 'B', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_rook': 'r', 'black_knight': 'n',
    'black_bishop': 'b', 'black_queen': 'q', 'black_king': 'k',
}

def get_8x8_board_from_frame(frame):
    """Use your exact board_vision.py functions to get FEN-style 8x8 board."""
    src_points = get_board_corners(frame)
    rectified, H, side_px = rectify_board(frame, src_points)
    square_size_px = side_px / 8.0

    # Your original template matching
    board_state_names = detect_pieces(rectified, square_size_px)

    # Convert to standard FEN lettering
    board_fen = [['.' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            name = board_state_names[r][c]
            board_fen[r][c] = PIECE_MAP.get(name, '.')
    return board_fen, rectified

def detect_move_uci(board_before, board_after):
    """Detect single move difference and return UCI (e.g. e2e4)."""
    from_sq = to_sq = None
    files = 'abcdefgh'
    ranks = '87654321'  # row 0 = rank 8 (top of image)

    for r in range(8):
        for c in range(8):
            if board_before[r][c] != board_after[r][c]:
                if board_before[r][c] != '.' and board_after[r][c] == '.':
                    from_sq = (r, c)
                elif board_before[r][c] == '.' and board_after[r][c] != '.':
                    to_sq = (r, c)
                elif board_before[r][c] != '.' and board_after[r][c] != '.':
                    from_sq = (r, c)  # capture
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

    print("\n=== Chess Vision Move Detector (Live Camera) ===")
    print("Instructions:")
    print("  'b' - Capture BEFORE board state")
    print("  'a' - Capture AFTER board state (after physical move)")
    print("  'q' - Quit\n")

    board_before = None
    before_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Live Camera Feed - Press b / a / q", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('b') and not before_captured:
            print("Capturing BEFORE board...")
            board_before, rectified_before = get_8x8_board_from_frame(frame)
            cv2.imwrite("before.jpg", frame)
            cv2.imwrite("debug_rectified_before.jpg", rectified_before)
            print("✅ BEFORE board captured successfully.")
            before_captured = True

        elif key == ord('a') and before_captured:
            print("Capturing AFTER board...")
            board_after, rectified_after = get_8x8_board_from_frame(frame)
            cv2.imwrite("after.jpg", frame)
            cv2.imwrite("debug_rectified_after.jpg", rectified_after)

            move_uci = detect_move_uci(board_before, board_after)

            print("\n=== FINAL 8x8 BOARD STATE (FEN letters) ===")
            for row in board_after:
                print(" ".join(row))

            print(f"\n=== DETECTED MOVE (UCI) ===\n{move_uci or 'No move detected (check for multiple changes or lighting issues)'}")

            # Reset for next capture pair
            before_captured = False
            board_before = None
            print("\n✅ Ready for next BEFORE capture (press 'b')...\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()