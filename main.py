"""
main.py — 6DOF Chess Bot Orchestrator
--------------------------------------
Master control loop for the chess-playing robot.
Press ENTER to trigger a photo capture cycle.
Press Ctrl+C or type 'halt' to stop.

Folder structure:
    <repo_root>/
        main.py                         ← this file
        ChessStuff/
            board_vision.py             — get_board_corners(), rectify_board(), detect_pieces()
            vision_move_detector.py     — get_8x8_board_from_frame(), detect_move_uci()
            stockfish_interface.py      — configure_engine(), apply_player_move(), get_engine_response()
        RobotStuff/
            RobotControl/
                path_planning.py        — execute_robot_move(move: str)
"""

import sys
import os
import threading
import cv2
import chess
import chess.engine

# ---------------------------------------------------------------------------
# Path setup — ensures sibling packages are importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
# vision_move_detector imports board_vision internally — we only need these two functions
from ChessStuff.vision_move_detector import get_8x8_board_from_frame, detect_move_uci

# stockfish_interface helpers
from ChessStuff.stockfish_interface import configure_engine, apply_player_move, get_engine_response

# TODO: Replace 'execute_robot_move' with the actual function name inside path_planning.py
from RobotStuff.RobotControl.path_planning import execute_robot_move  # (move: str) -> None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# TODO: Replace with the actual path to your Stockfish binary.
#       Linux/Mac default:  "/usr/bin/stockfish"
#       Windows example:    r"C:\Users\YourName\Downloads\stockfish\stockfish.exe"
STOCKFISH_PATH = r"C:\Users\tkubi\Documents\Programs\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# Strength setting — valid range 1320–3190 ELO
ENGINE_ELO    = 1500

# Seconds Stockfish is allowed to think per move
THINK_TIME    = 1.0

# Which camera index to use (0 = default webcam)
CAMERA_INDEX  = 0

# ---------------------------------------------------------------------------
# Game state  (module-level so every cycle shares the same objects)
# ---------------------------------------------------------------------------
board: chess.Board                         = chess.Board()       # tracks full move history
engine: chess.engine.SimpleEngine | None   = None                # opened once, reused each cycle
board_before: list[list[str]] | None       = None                # 8×8 FEN grid from previous photo
halt_flag                                  = threading.Event()

# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {CAMERA_INDEX}. "
            "Check CAMERA_INDEX in the config section."
        )
    return cap


def open_engine() -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    configure_engine(eng, ENGINE_ELO)
    return eng


def print_board(board_grid: list[list[str]], label: str = "Board state"):
    """Pretty-print an 8×8 FEN-letter grid with rank/file labels."""
    print(f"\n  === {label} ===")
    print("    a b c d e f g h")
    print("    ---------------")
    for rank_idx, row in enumerate(board_grid):
        rank_number = 8 - rank_idx
        print(f"  {rank_number}| {' '.join(row)}")
    print()

# ---------------------------------------------------------------------------
# Core cycle — triggered by pressing ENTER
# ---------------------------------------------------------------------------

def run_cycle(cap: cv2.VideoCapture):
    """
    One full capture → detect → stockfish → robot cycle.
    Mutates module-level: board_before, board, (engine is read-only here).
    """
    global board_before, board, engine

    # ── Step 1: Capture current frame ────────────────────────────────────────
    print("\n[1/5] Capturing frame from camera...")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed. Check connection and CAMERA_INDEX.")
    print("      Frame captured OK.")

    # ── Step 2: Detect piece locations on the board ───────────────────────────
    print("[2/5] Detecting piece locations...")
    board_after, rectified = get_8x8_board_from_frame(frame)
    print_board(board_after, label="Current board (all pieces)")

    # First cycle — just store baseline, nothing to compare yet
    if board_before is None:
        print("      No previous snapshot — storing as baseline.")
        print("      Make the human's first move on the board, then press ENTER.")
        board_before = board_after
        return

    # ── Step 3: Detect the most recent player move ────────────────────────────
    print("[3/5] Detecting most recent move...")
    uci_move = detect_move_uci(board_before, board_after)

    if uci_move is None:
        print("      No move detected. Board may not have changed — skipping cycle.")
        print("      If a move was made, check lighting or template matching threshold.")
        return

    print(f"      Detected player move (UCI): {uci_move}")

    # Convert UCI → SAN so python-chess / stockfish_interface can consume it
    try:
        move_obj  = chess.Move.from_uci(uci_move)
        san_move  = board.san(move_obj)        # needs pre-move board state
    except (ValueError, chess.InvalidMoveError) as exc:
        print(f"      [ERROR] UCI move '{uci_move}' is not legal in current position: {exc}")
        print("      Board state may be out of sync. Resetting baseline and retrying.")
        board_before = board_after
        return

    print(f"      Move in SAN notation: {san_move}")

    # Advance our internal python-chess board with the player's move
    if not apply_player_move(board, san_move):
        print(f"      [ERROR] apply_player_move() rejected '{san_move}'. Skipping cycle.")
        board_before = board_after
        return

    # Update baseline AFTER a confirmed move
    board_before = board_after

    # Check for game-over after player's move
    if board.is_game_over():
        print(f"\n[GAME OVER] Result: {board.result()}")
        halt_flag.set()
        return

    # ── Step 4: Query Stockfish for best reply ────────────────────────────────
    print("[4/5] Querying Stockfish...")
    stockfish_san = get_engine_response(board, engine)   # also pushes move onto `board`
    print(f"      Stockfish replies (SAN): {stockfish_san}")

    # Convert Stockfish's reply back to UCI for the robot
    # get_engine_response() already pushed the move, so we need the last move on the stack
    stockfish_uci = board.peek().uci()
    print(f"      Stockfish reply (UCI):   {stockfish_uci}")

    # Check for game-over after engine's move
    if board.is_game_over():
        print(f"\n[GAME OVER] Result: {board.result()}")

    # ── Step 5: Send move to robot arm ────────────────────────────────────────
    print("[5/5] Executing robot move...")
    # TODO: execute_robot_move receives the UCI string (e.g. 'e7e5').
    #       If your path_planning.py expects a different format (e.g. board
    #       coordinates, SAN, or a move object), convert stockfish_uci here first.
    execute_robot_move(stockfish_uci)

    print("      Cycle complete. Waiting for next trigger.\n")
    print(f"  Full move history so far: {board.move_stack}\n")

# ---------------------------------------------------------------------------
# Input listener — runs on a background thread
# ---------------------------------------------------------------------------

def input_listener(cap: cv2.VideoCapture):
    """
    Listens for keyboard input on a background thread.
    ENTER  → triggers a capture cycle
    'halt' → sets halt_flag to stop the main loop
    """
    print("  Press ENTER after each human move to capture and respond.")
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

        try:
            run_cycle(cap)
        except Exception as exc:
            print(f"[ERROR] Cycle failed: {exc}")
            print("        Ready for next trigger.\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global engine

    print("=" * 52)
    print("  6DOF Chess Bot — Master Controller")
    print("=" * 52)

    # Open camera and engine once; share across all cycles
    print("Opening camera...")
    cap = open_camera()
    print(f"  Camera index {CAMERA_INDEX} opened OK.")

    print(f"Opening Stockfish engine at: {STOCKFISH_PATH}")
    engine = open_engine()
    print(f"  Engine opened OK (ELO cap: {ENGINE_ELO}).\n")

    print("New game — standard starting position.")
    print("Press ENTER once with pieces in starting position to capture the baseline.")
    print("Then make the first human move and press ENTER again.\n")

    listener = threading.Thread(target=input_listener, args=(cap,), daemon=True)
    listener.start()

    try:
        halt_flag.wait()          # Block main thread until halt is requested
    except KeyboardInterrupt:
        print("\n[HALT] Ctrl+C received. Shutting down.")
        halt_flag.set()
    finally:
        cap.release()
        if engine:
            engine.quit()
        print("Camera and engine closed. Goodbye.")


if __name__ == "__main__":
    main()