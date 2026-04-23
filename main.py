"""
main.py — 6DOF Chess Bot Orchestrator
--------------------------------------
Master control loop for the chess-playing robot.
Press ENTER to trigger a photo capture cycle.
Press Ctrl+C or type 'halt' to stop.

Folder structure assumed:
    ChessStuff/
        board_vision.py         — capture_photo()
        vision_move_detector.py — detect_move(prev_path, curr_path) -> str | None
    RobotStuff/
        RobotControl/
            path_planning.py    — execute_robot_move(move: str)
Stockfish integration is called via a subprocess or wrapper defined below.

TODO: Before running, work through all comments marked TODO in this file.
      They mark every place that needs to be wired up to your actual scripts.
"""

import sys
import os
import threading
import subprocess
import importlib
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — ensures sibling packages are importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
# TODO: Replace 'capture_photo' with the actual function name inside board_vision.py
#       that triggers your camera and saves an image.
from ChessStuff.board_vision         import capture_photo          # -> str (saved image path)

# TODO: Replace 'detect_move' with the actual function name inside vision_move_detector.py
#       that compares two images and returns a move string (or None if no move).
from ChessStuff.vision_move_detector import detect_move            # (prev, curr) -> str | None

# TODO: Replace 'execute_robot_move' with the actual function name inside path_planning.py
#       that takes a move string and physically moves the robot arm.
from RobotStuff.RobotControl.path_planning import execute_robot_move  # (move: str) -> None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PHOTO_DIR        = os.path.join(BASE_DIR, "photos")

# TODO: Replace this path with the actual path to your Stockfish executable.
#       e.g. r"C:\Users\YourName\Downloads\stockfish\stockfish-windows-x86-64.exe"
STOCKFISH_PATH   = r"C:\path\to\stockfish.exe"

# TODO: Adjust search depth if needed. Higher = stronger but slower (15 is a solid default).
STOCKFISH_DEPTH  = 15

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
previous_photo_path: str | None = None
halt_flag = threading.Event()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamped_path() -> str:
    """Returns a unique file path for each captured photo."""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PHOTO_DIR, f"board_{stamp}.png")


def query_stockfish(move: str) -> str:
    """
    Send a detected player move to Stockfish and get its response move.
    Uses Stockfish's UCI protocol via subprocess.

    Args:
        move: The player's move in UCI format, e.g. 'e2e4'

    Returns:
        Stockfish's best reply in UCI format, e.g. 'e7e5'
    """
    commands = "\n".join([
        "uci",
        "isready",
        # TODO: This currently only sends the most recent move to Stockfish, which means
        #       it always evaluates from the starting position + 1 move.
        #       For a full game, you'll need to track all moves and pass the complete
        #       move history here, e.g: "position startpos moves e2e4 e7e5 g1f3 ..."
        f"position startpos moves {move}",
        f"go depth {STOCKFISH_DEPTH}",
    ])

    result = subprocess.run(
        [STOCKFISH_PATH],
        input=commands,
        capture_output=True,
        text=True,
        timeout=30,
    )

    for line in reversed(result.stdout.splitlines()):
        if line.startswith("bestmove"):
            return line.split()[1]   # e.g. 'e7e5'

    raise RuntimeError(f"Stockfish did not return a best move.\nOutput:\n{result.stdout}")


# ---------------------------------------------------------------------------
# Core cycle — triggered by 'take photo' input
# ---------------------------------------------------------------------------

def run_cycle():
    """
    One full capture-detect-respond-execute cycle.
    Mutates global previous_photo_path.
    """
    global previous_photo_path

    print("\n[1/5] Capturing photo...")
    # TODO: Check what arguments capture_photo() expects in board_vision.py.
    #       Currently it is passed a file path to save to. If your function
    #       handles its own save path internally, remove the argument entirely:
    #       current_photo_path = capture_photo()
    current_photo_path = capture_photo(timestamped_path())
    print(f"      Saved to: {current_photo_path}")

    if previous_photo_path is None:
        print("[2/5] No previous photo — this is the baseline. Waiting for next trigger.")
        previous_photo_path = current_photo_path
        return

    print("[2/5] Comparing photos to detect move...")
    # TODO: Check what detect_move() returns in vision_move_detector.py.
    #       This assumes it returns a move in UCI format (e.g. 'e2e4') or None.
    #       If it returns a different format, you may need to convert it before
    #       passing it to query_stockfish() below.
    player_move = detect_move(previous_photo_path, current_photo_path)

    if player_move is None:
        print("      No move detected. Board may not have changed. Skipping cycle.")
        return

    print(f"      Move detected: {player_move}")
    previous_photo_path = current_photo_path

    print("[3/5] Querying Stockfish...")
    stockfish_move = query_stockfish(player_move)
    print(f"      Stockfish replies: {stockfish_move}")

    print("[4/5] Executing robot move...")
    # TODO: Check what execute_robot_move() expects in path_planning.py.
    #       Currently it is passed the raw UCI move string from Stockfish (e.g. 'e7e5').
    #       If your function expects a different format (e.g. joint angles, board
    #       coordinates, or a move object), convert stockfish_move here first.
    execute_robot_move(stockfish_move)

    print("[5/5] Cycle complete. Waiting for next trigger.\n")


# ---------------------------------------------------------------------------
# Input listener — runs on a background thread
# ---------------------------------------------------------------------------

def input_listener():
    """
    Listens for keyboard input on a background thread.
    ENTER  → triggers a capture cycle
    'halt' → sets halt_flag to stop the main loop
    """
    print("  Press ENTER to capture a photo and run a cycle.")
    print("  Type 'halt' and press ENTER (or Ctrl+C) to stop.\n")

    while not halt_flag.is_set():
        try:
            user_input = input()
        except EOFError:
            break

        if user_input.strip().lower() == "halt":
            print("[HALT] Stop requested via input.")
            halt_flag.set()
            break

        # Any other input (including bare ENTER) triggers a cycle
        try:
            run_cycle()
        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            print("        Ready for next trigger.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  6DOF Chess Bot — Master Controller")
    print("=" * 50)

    listener = threading.Thread(target=input_listener, daemon=True)
    listener.start()

    try:
        halt_flag.wait()          # Block main thread until halt is requested
    except KeyboardInterrupt:
        print("\n[HALT] Ctrl+C received. Shutting down.")
        halt_flag.set()

    print("Goodbye.")


if __name__ == "__main__":
    main()
