import chess
import chess.engine
import sys

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
STOCKFISH_PATH = "/usr/bin/stockfish"        # Linux/Mac
# STOCKFISH_PATH = "C:/stockfish/stockfish.exe"  # Windows

ENGINE_ELO     = 1500   # Set anywhere from 1320 to 3190
THINK_TIME     = 1.0    # Seconds Stockfish spends thinking per move
# ---------------------------------------------------------------


def configure_engine(engine: chess.engine.SimpleEngine, elo: int):
    """Enable ELO-limited strength mode."""
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": max(1320, min(3190, elo))  # Clamp to valid range
    })


def get_engine_response(board: chess.Board, engine: chess.engine.SimpleEngine) -> str:
    """
    Ask the engine for its best move given the current board state.
    Returns the move in SAN notation (e.g. 'Nd5', 'Nxd5', 'O-O', 'e8=Q').
    """
    result = engine.play(board, chess.engine.Limit(time=THINK_TIME))
    
    # Convert the engine's move to SAN *before* pushing it to the board
    # (SAN requires the board to be in the pre-move state)
    san_move = board.san(result.move)
    board.push(result.move)
    
    return san_move


def apply_player_move(board: chess.Board, san: str) -> bool:
    """
    Parse and apply a player move given in SAN notation.
    Returns True on success, False if the move is invalid.
    """
    try:
        move = board.parse_san(san)
        board.push(move)
        return True
    except ValueError:
        return False


def main():
    # Expect the player's SAN move as a command-line argument
    # Usage: python chess_engine.py "Nd5"
    if len(sys.argv) < 2:
        print("Usage: python chess_engine.py <SAN_move>")
        print("Example: python chess_engine.py Nd5")
        sys.exit(1)

    player_move_san = sys.argv[1]

    board  = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    configure_engine(engine, ENGINE_ELO)

    try:
        # Apply the incoming player move
        if not apply_player_move(board, player_move_san):
            print(f"ERROR: '{player_move_san}' is not a legal SAN move in this position.")
            sys.exit(1)

        # Check if the game is already over after the player's move
        if board.is_game_over():
            print(f"GAME_OVER: {board.result()}")
            sys.exit(0)

        # Get and print the engine's response
        engine_move_san = get_engine_response(board, engine)
        print(engine_move_san)   # e.g. "Nxd5" — clean output for your robot script to consume

    finally:
        engine.quit()


if __name__ == "__main__":
    main()