"""
main.py — 6DOF Chess Bot (Robust Ctrl+C Fix v2)
Guaranteed clean shutdown even when OpenCV + threads are involved.
"""

import sys
import os
import threading
import signal
import time
import atexit
import cv2
import chess
import chess.engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ChessStuff.vision_move_detector import (
    get_8x8_board_from_frame,
    detect_move_uci,
    get_good_frame
)

from ChessStuff.stockfish_interface import configure_engine, apply_player_move, get_engine_response
from RobotStuff.RobotControl.path_planning import execute_robot_move

STOCKFISH_PATH = r"C:\Users\tkubi\Documents\Programs\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
ENGINE_ELO = 1500
CAMERA_INDEX = 0

board: chess.Board = chess.Board()
engine = None
board_before = None
halt_flag = threading.Event()
cap = None


def force_cleanup():
    """Nuclear but reliable cleanup for OpenCV + Windows."""
    global cap, engine
    print("\n[FORCE CLEANUP] Shutting down...")

    # 1. Stop camera
    if cap is not None:
        try:
            cap.release()
        except:
            pass

    # 2. Quit engine
    if engine is not None:
        try:
            engine.quit()
        except:
            pass

    # 3. Destroy ALL OpenCV windows (do it multiple times - sometimes needed)
    for _ in range(3):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass

    time.sleep(0.3)          # Give Windows time to release resources
    print("[FORCE CLEANUP] Done. Exiting now.")
    os._exit(0)              # Hard exit - bypasses normal Python shutdown


# Register cleanup for all exit paths
atexit.register(force_cleanup)


def signal_handler(sig, frame):
    print("\n[Ctrl+C] Caught interrupt signal.")
    halt_flag.set()
    force_cleanup()


signal.signal(signal.SIGINT, signal_handler)


def open_camera():
    global cap
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    return cap


def open_engine():
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    configure_engine(eng, ENGINE_ELO)
    return eng


def print_board(board_grid, label="Board state"):
    print(f"\n  === {label} ===")
    print("    a b c d e f g h")
    print("    ---------------")
    for i, row in enumerate(board_grid):
        print(f"  {8-i}| {' '.join(row)}")
    print()


def run_cycle(cap):
    global board_before, board, engine

    print("\n[1/5] Capturing HDR + cropped frame...")
    frame = get_good_frame(cap)
    if frame is None:
        return
    print("      Frame ready.")

    print("[2/5] Detecting pieces...")
    board_after, rectified, debug_frame = get_8x8_board_from_frame(frame)

    cv2.namedWindow("Debug Corners", cv2.WINDOW_NORMAL)
    cv2.imshow("Debug Corners", debug_frame)
    cv2.waitKey(1)

    print_board(board_after, "Current board")

    if board_before is None:
        print("      Baseline stored. Make your move and press ENTER.")
        board_before = board_after
        return

    print("[3/5] Detecting move...")
    uci_move = detect_move_uci(board_before, board_after)
    if uci_move is None:
        print("      No move detected.")
        return
    print(f"      Move: {uci_move}")

    try:
        move_obj = chess.Move.from_uci(uci_move)
        san_move = board.san(move_obj)
    except Exception as e:
        print(f"      Invalid move: {e}")
        board_before = board_after
        return

    if not apply_player_move(board, san_move):
        board_before = board_after
        return

    board_before = board_after

    if board.is_game_over():
        print(f"\n[GAME OVER] {board.result()}")
        halt_flag.set()
        return

    print("[4/5] Stockfish thinking...")
    stockfish_san = get_engine_response(board, engine)
    stockfish_uci = board.peek().uci()
    print(f"      Stockfish: {stockfish_san} ({stockfish_uci})")

    print("[5/5] Sending move to robot...")
    execute_robot_move(stockfish_uci)
    print("      Cycle complete.\n")


def input_listener(cap):
    print("Press ENTER after each human move. Type 'halt' to stop.\n")
    while not halt_flag.is_set():
        try:
            user_input = input()
        except EOFError:
            break
        if user_input.strip().lower() == "halt":
            halt_flag.set()
            break
        try:
            run_cycle(cap)
        except Exception as e:
            print(f"[ERROR] {e}")


def main():
    global engine, cap

    print("=" * 52)
    print("  6DOF Chess Bot — Master Controller (Robust Shutdown)")
    print("=" * 52)

    cap = open_camera()
    engine = open_engine()

    print("Camera and Stockfish ready.\n")
    print("Press ENTER with starting position to begin.\n")

    listener = threading.Thread(target=input_listener, args=(cap,), daemon=False)
    listener.start()

    try:
        halt_flag.wait()
    except KeyboardInterrupt:
        pass
    finally:
        force_cleanup()


if __name__ == "__main__":
    main()