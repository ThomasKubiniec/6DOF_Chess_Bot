#!/usr/bin/env python3
"""
Interactive Chess Piece Template Capture Tool
Leverages existing HDR fusion (make_hdr.py), board rectification & corner detection (board_vision.py),
and cropping logic from the project.

Usage:
  python capture_piece_template.py

Workflow:
  1. Live camera feed runs (fast raw preview).
  2. Press 'c' → captures HDR-fused + rectified board view.
  3. Drag a tight rectangle around ONE chess piece in the rectified window.
  4. Press ENTER/SPACE to confirm crop.
  5. Type a descriptive name in the terminal (e.g. white_pawn_07 or black_knight_02).
  6. Image is saved as PNG to the piece_templates folder.
  7. Repeat as needed. Press 'q' in live window to quit.

This produces high-quality templates that match the exact coordinate system
used by detect_pieces() in board_vision.py → perfect for robust template matching.
"""

import cv2
import numpy as np
import os
import sys
import time

# ================== ROBUST PATH SETUP (same pattern as vision_move_detector.py) ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print("=== Interactive Piece Template Capture ===")
print(f"Project root: {project_root}")

# Try to import everything we need
get_hdr_chessboard = None
get_board_corners = None
rectify_board = None
TEMPLATES_DIR = None
CROP_LEFT_PX = 100
CROP_RIGHT_PX = 80
CAMERA_INDEX = 0

try:
    from make_hdr import get_hdr_chessboard as _get_hdr
    get_hdr_chessboard = _get_hdr
    print("✅ HDR fusion imported from make_hdr.py")
except Exception as e:
    print(f"⚠️  HDR not available: {e}")

try:
    from ChessStuff.board_vision import (
        get_board_corners as _get_corners,
        rectify_board as _rectify,
        TEMPLATES_DIR as _templates_dir,
        CROP_LEFT_PX as _crop_l,
        CROP_RIGHT_PX as _crop_r,
        CAMERA_INDEX as _cam
    )
    get_board_corners = _get_corners
    rectify_board = _rectify
    TEMPLATES_DIR = _templates_dir
    CROP_LEFT_PX = _crop_l
    CROP_RIGHT_PX = _crop_r
    CAMERA_INDEX = _cam
    print("✅ Board vision functions imported")
except Exception as e:
    print(f"⚠️  Using fallback paths (board_vision import failed: {e})")
    # Fallback for standalone use
    TEMPLATES_DIR = os.path.join(project_root, "ChessStuff", "piece_templates")
    if not os.path.exists(TEMPLATES_DIR):
        TEMPLATES_DIR = os.path.join(project_root, "piece_templates")
    if not os.path.exists(TEMPLATES_DIR):
        TEMPLATES_DIR = os.path.join(os.getcwd(), "piece_templates")

print(f"📁 Templates will be saved to: {TEMPLATES_DIR}")
os.makedirs(TEMPLATES_DIR, exist_ok=True)


def get_good_frame(cap):
    """Capture HDR-fused + margin-cropped frame (reuses project logic)."""
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


def main():
    print("\n=== INSTRUCTIONS ===")
    print("• Live window shows fast camera preview")
    print("• Press 'c' to capture an HDR + rectified snapshot")
    print("• In the rectified window: click and drag a tight box around ONE piece")
    print("• Press ENTER or SPACE to accept the crop")
    print("• Type a name like 'white_pawn_05' or 'black_knight_02' and press ENTER")
    print("• Repeat for as many variants as you want (aim for 8–12 per piece type)")
    print("• Press 'q' in the live window to quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera at index", CAMERA_INDEX)
        return

    cv2.namedWindow("Live Preview - Press 'c' to capture | 'q' to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Preview - Press 'c' to capture | 'q' to quit", 800, 600)

    while True:
        # Fast live preview (raw camera — responsive)
        ret, live = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Apply same side crop for visual consistency
        h, w = live.shape[:2]
        live = live[:, CROP_LEFT_PX : w - CROP_RIGHT_PX].copy()

        # Overlay instructions
        cv2.putText(live, "Press 'c' = capture HDR + rectified (tall pieces now fully visible)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(live, "Press 'q' = quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Live Preview - Press 'c' to capture | 'q' to quit", live)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            print("\n[CAPTURE] Taking HDR snapshot + rectifying board...")
            good_frame = get_good_frame(cap)
            if good_frame is None:
                print("  ❌ Failed to get good frame — using live frame instead")
                good_frame = live.copy()

            # Rectify with EXTRA expansion + slightly larger canvas so tall corner pieces
            # (kings/queens) are fully visible for template capture.
            try:
                src_points, debug = get_board_corners(good_frame)

                # More aggressive expansion just for template capture (main detection still uses 1.15)
                center = src_points.mean(axis=0)
                expanded = [center + (pt - center) * 1.18 for pt in src_points]
                src_points_exp = np.float32(expanded)

                # Use 880 px output (10% larger than 800) → gives breathing room for tall pieces
                rectified, H, side_px = rectify_board(good_frame, src_points_exp, output_size=880)
                print(f"  ✅ Board rectified ({side_px}x{side_px} px) — extra room for tall pieces")

                # Show rectified board and let user select the piece
                select_win = "Rectified Board — Drag tight box around ONE piece (tall pieces fully visible)"
                cv2.namedWindow(select_win, cv2.WINDOW_NORMAL)
                cv2.imshow(select_win, rectified)

                print("  → Drag rectangle around the target piece, then press ENTER/SPACE")
                roi = cv2.selectROI(select_win, rectified, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow(select_win)

                if roi[2] <= 10 or roi[3] <= 10:
                    print("  ⚠️  Invalid or cancelled selection — try again")
                    continue

                x, y, w_roi, h_roi = roi
                cropped = rectified[y : y + h_roi, x : x + w_roi].copy()

                # Preview the crop
                preview_win = "Cropped Template Preview (close window when ready)"
                cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
                cv2.imshow(preview_win, cropped)
                cv2.waitKey(300)  # brief pause so user sees it

                # Get filename from terminal
                suggested = "white_pawn_XX" if "white" in str(roi) else "black_piece_XX"
                name = input(f"  Enter template filename (no extension, e.g. {suggested}): ").strip()
                if not name:
                    print("  ⚠️  No name given — discarding crop")
                    cv2.destroyWindow(preview_win)
                    continue

                if not name.lower().endswith(".png"):
                    name += ".png"

                save_path = os.path.join(TEMPLATES_DIR, name)
                cv2.imwrite(save_path, cropped)
                print(f"  ✅ SAVED: {save_path}  ({w_roi}x{h_roi} px)")

                cv2.destroyWindow(preview_win)

                # Quick verification
                loaded = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    print(f"     Verified load: {loaded.shape} (grayscale ready for matching)")

            except RuntimeError as e:
                print(f"  ❌ Board detection failed: {e}")
                print("     Tip: Improve lighting / align camera better and try again")
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Template capture session ended.")
    print(f"   New templates are in: {TEMPLATES_DIR}")
    print("   Restart board_vision.py or your bot to use them.")


if __name__ == "__main__":
    main()