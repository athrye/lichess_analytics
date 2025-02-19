import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger

pd.set_option('display.max_columns', 10)


def parse_clock(comment: str) -> Optional[str]:
    """
    Given the entire text inside { ... }, tries to find something like [%clk 0:03:00].
    Returns the clock time string (e.g. '0:03:00') if found, else None.
    """
    match = re.search(r"\[%clk\s+([^\]]+)\]", comment)
    return match.group(1) if match else None


def read_pgn_file_into_games(
        pgn_path: Union[str, Path]
) -> list[dict]:
    """
    Reads the entire PGN file and returns a list of games.
    Each 'game' is a dict with:
      - 'headers': dict of all [Header "Value"] pairs
      - 'moves':  a single string containing the moves portion
    We detect a new game by lines that start with '[Event ' or any '[X ' header.

    We do NOT parse moves in this function. We just isolate:
        (header lines) + (move text lines)
    to form each game’s data for further parsing.
    """
    games = []
    with open(pgn_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        # Skip empty lines until we find a line that looks like `[Event "something"]`
        if not line:
            i += 1
            continue

        if line.startswith("[") and '"' in line:
            # Start of a new game
            headers = {}
            # 1) Read header lines
            while i < n and lines[i].strip().startswith("["):
                hdr_line = lines[i].strip()
                i += 1

                # Example: [Event "Rated blitz game"]
                match = re.match(r'^\[([^\]]+)\s+"([^"]+)"\]$', hdr_line)
                if match:
                    hdr_key = match.group(1)
                    hdr_val = match.group(2)
                    headers[hdr_key] = hdr_val

            # 2) Now read move lines until we see the next `[Header ` or EOF
            move_lines = []
            while i < n:
                if lines[i].strip().startswith("[") and '"' in lines[i]:
                    # This indicates a new game's header line => stop reading moves
                    break
                move_lines.append(lines[i])
                i += 1

            moves_str = "".join(move_lines).strip()

            # Store this game
            games.append({
                "headers": headers,
                "moves": moves_str,
            })
        else:
            # This line isn't a header line—maybe leftover or empty. Skip it.
            i += 1

    return games


def remove_variations(text: str) -> str:
    """
    Removes anything in parentheses (...) including nested, if you want to be thorough.
    For a simpler approach, we can just remove top-level parentheses with a naive regex.
    """
    # A naive approach: remove everything from '(' to the matching ')'
    # For nested parentheses, you'd need a stack-based approach or a more complex regex.
    # But for typical PGN, a single level is often enough. We'll do a simple approach:
    return re.sub(r"\(.*?\)", "", text, flags=re.DOTALL)


def tokenize_moves(move_text: str) -> list[str]:
    """
    Given a PGN moves string (without parentheses),
    splits on whitespace to produce tokens. E.g.:

    "1. e4 { [%clk 0:03:00] } 1... d5 ..."

    => ["1.", "e4", "{", "[%clk", "0:03:00]", "}", "1...", "d5", ...]
    """
    # Normalize newlines/spaces
    move_text = move_text.replace("\n", " ")
    # Also remove double or triple spaces
    move_text = re.sub(r"\s+", " ", move_text)
    tokens = move_text.strip().split(" ")
    return [t for t in tokens if t]  # remove empty


def parse_game_moves(
        headers: dict,
        moves_str: str
) -> list[dict]:
    # (1) Remove parentheses.
    moves_str = remove_variations(moves_str)
    # (2) Tokenize into pieces (numbers, moves, braces, etc.).
    tokens = tokenize_moves(moves_str)

    rows = []
    move_number = 0
    color = "white"  # first move is always White
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]

        # (A) Check for move-number tokens like "7." or "7...".
        m_white = re.match(r"^(\d+)\.$", token)
        m_black = re.match(r"^(\d+)\.\.\.$", token)
        if m_white:
            move_number = int(m_white.group(1))
            color = "white"
            i += 1
            continue
        elif m_black:
            move_number = int(m_black.group(1))
            color = "black"
            i += 1
            continue

        # (B) If token is a game‐ending string like "1-0" or "0-1", stop.
        if token in ["1-0", "0-1", "1/2-1/2", "*"]:
            break

        # (C) If it’s a “normal” move token (like "e4", "Nf6", "Bxf3?!", etc.),
        #     read the move, then gather *all* comment blocks that follow immediately.
        possible_move = token
        clock_time = None
        i += 1  # advance past the move token

        # Gather *all* consecutive `{ ... }` blocks in a row
        while i < n and tokens[i] == "{":
            i += 1  # skip '{'
            comment_tokens = []
            while i < n and tokens[i] != "}":
                comment_tokens.append(tokens[i])
                i += 1
            i += 1  # skip the closing '}'

            # Parse clock from this comment block
            comment_str = " ".join(comment_tokens)
            found_clock = parse_clock(comment_str)
            if found_clock:
                clock_time = found_clock
                # If you want the *last* clock time to override earlier ones, do nothing else here
                # If you want the *first* one only, you could “break” once you find it.

        # Now we have a move, a move number, a color, and possibly a clock_time.
        row_dict = dict(headers)
        row_dict["MoveNumber"] = move_number
        row_dict["Color"] = color
        row_dict["Move"] = possible_move
        row_dict["Clock"] = clock_time
        rows.append(row_dict)

        # Flip color for the next token (white→black, black→white),
        # because PGN typically goes "1. e4 1... d5" for the same move number.
        if color == "white":
            color = "black"
        else:
            color = "white"

    return rows


def pgn_to_dataframe(pgn_path: Union[str, Path]) -> pd.DataFrame:
    """
    High-level function:
      1) Reads the entire PGN file into a list of games (headers + moves text).
      2) For each game, parse out main-line moves ignoring variations.
      3) Combine into a single DataFrame with one row per move, repeating headers.
    """
    games_data = read_pgn_file_into_games(pgn_path)

    all_rows = []
    for game in games_data:
        headers = game["headers"]
        moves_str = game["moves"]
        rows_for_this_game = parse_game_moves(headers, moves_str)
        all_rows.extend(rows_for_this_game)

    df = pd.DataFrame(all_rows)
    return df


def filter_games_by_required_moves(
        df: pd.DataFrame,
        required_moves: list[dict],
        group_col: str = "GameId"
) -> pd.DataFrame:
    """
    Filters `df` so we keep only games (groups) that satisfy *all* of the
    specified move conditions in `required_moves`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per move. Must contain `group_col` (e.g. GameId),
        plus columns like "MoveNumber", "Color", "Move", etc.
    required_moves : list of dict
        Each dict is a set of conditions that must match at least once in the game.
        For example:
            [
              {"MoveNumber": 1, "Color": "white", "Move": "e4"},
              {"MoveNumber": 1, "Color": "black", "Move": "d5"},
            ]
        means each game must have:
          - a row with (MoveNumber=1, Color="white", Move="e4"), AND
          - a row with (MoveNumber=1, Color="black", Move="d5")
        to be retained.
    group_col : str
        Column name to group the data by (often "GameId").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the moves from games that satisfy all conditions.
    """

    def group_satisfies_all_conditions(game_df: pd.DataFrame) -> bool:
        """
        Returns True if this group has *at least one row* matching each
        condition in `required_moves`.
        """
        for cond in required_moves:
            # Convert dict to a boolean mask
            mask = pd.Series([True] * len(game_df), index=game_df.index)

            # For each key in cond, refine the mask
            for col_key, required_val in cond.items():
                # If the game doesn't have that column, the condition fails
                if col_key not in game_df.columns:
                    return False
                mask = mask & (game_df[col_key] == required_val)

            # Now we check if *any* row matches this entire condition
            if not mask.any():
                return False

        return True

    # Group by the game ID column and apply the check
    grouped = df.groupby(group_col, group_keys=False)
    filtered_df = grouped.filter(group_satisfies_all_conditions)
    return filtered_df


def hhmmss_to_sec(s: str) -> float:
    hh, mm, ss = s.split(':')
    return ss + mm * 60 + hh * 3600


if __name__ == "__main__":

    # Extract for Scandinavian
    scandi_conditions = [
        {"MoveNumber": 1, "Color": "white", "Move": "e4"},
        {"MoveNumber": 1, "Color": "black", "Move": "d5"},
    ]
    player_name: str = "athrye"
    player_color: str = 'Black'
    drop_correspndence_games: bool = True
    drop_increment: int = True

    # I/O
    target_path: Path = Path.cwd() / 'data' / 'inputs' / 'athrye_data_20250217.pgn'
    output_path: Path = Path.cwd() / 'data' / 'outputs' / 'athrye_data_20250217.csv'
    filtered_output_path: Path = Path.cwd() / 'data' / 'outputs' / 'athrye_data_20250217-scandi.csv'
    # Check file
    if not target_path.exists():
        raise FileNotFoundError(f"File not found: file://{target_path.absolute()}")

    # Convert pgn to dataframe
    df = pgn_to_dataframe(target_path)
    logger.info(f"Created dataframe with {len(df)} rows total")

    # Drop correspondence games
    if drop_correspndence_games:
        df = df[df['TimeControl'] != '-']
        logger.info(f'After removing correspondence games, have {len(df)} rows total')

    # Do some feature engineering
    df['ClockStartSec'] = [int(x.split('+')[0]) for x in df['TimeControl']]
    df['IncrementSec'] = [int(x.split('+')[1]) for x in df['TimeControl']]
    df['RemainingTimeSec'] = [hhmmss_to_sec(s_) for s_ in df['Clock']]
    df['StartMinusClock'] = df['ClockStartSec'] - df['RemainingTimeSec']

    # Drop increment games
    if drop_increment:
        df = df[df['IncrementSec'] == 0]
        logger.info(f"After removing increment games, have {len(df)} rows total")

    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Parsed all PGNs to DataFrame, head:\n{df.head()}")
    logger.info(f"Saved to {output_path}")

    # 3) Filter
    df_scandi = filter_games_by_required_moves(df, scandi_conditions, group_col="GameId")
    logger.info(f"Found {len(df_scandi)} rows of scandi in particular")
    df_scandi = df_scandi[df_scandi[player_color] == player_name]
    logger.info(f"After filtering for {player_color}={player_name}, have {len(df_scandi)} rows")
    logger.info(f"This corresponds to {df['GameId'].nunique()} games")

    # 4) Write out or inspect
    df_scandi.to_csv(filtered_output_path, index=False)
    logger.info(f"Saved to file://{filtered_output_path.absolute()}")
