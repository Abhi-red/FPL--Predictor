"""Nudge raw predictions using retrieved news - only on a clear, specific signal.

For every player with a raw prediction for the upcoming gameweek we retrieve
recent news. A chunk only counts if it mentions the player's surname AND a
categorised keyword:

  OUT   (ruled out / suspended / surgery / ...)  -> x(1 - ADJUSTMENT_CAP)
  DOUBT (knock / late test / rotation risk / ...) -> x0.85
  BOOST (back in training / expected to start ...) -> x(1 + ADJUSTMENT_CAP/2)

Anything else leaves the prediction untouched. The factor is always clamped to
[1 - ADJUSTMENT_CAP, 1 + ADJUSTMENT_CAP] - news can nudge a prediction, never
replace it. The factor + a short reason + the source URL are stored on
``predictions``.

Run (after predict.py and embed.py):
    python src/rag/adjust.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import ADJUSTMENT_CAP, SEASON  # noqa: E402
from db import get_connection  # noqa: E402

OUT_TERMS = (
    "ruled out", "sidelined", "will miss", "out for", "out of the", "suspended",
    "suspension", "ban ", "banned", "surgery", "long-term", "long term",
    "not in the squad", "won't play", "will not play", "months out", "acl",
)
DOUBT_TERMS = (
    "doubt", "knock", "assessed", "late test", "late fitness", "rotation risk",
    "may be rested", "could be rested", "75%", "50%", "minor", "slight",
    "hamstring tightness", "carrying a", "fitness test",
)
BOOST_TERMS = (
    "back in training", "returned to training", "expected to start", "set to start",
    "returns to the squad", "fit again", "available again", "back available",
    "passed a fitness test", "in contention",
)
CATEGORIES = (("OUT", OUT_TERMS), ("DOUBT", DOUBT_TERMS), ("BOOST", BOOST_TERMS))

CLAMP_LOW = 1.0 - ADJUSTMENT_CAP
CLAMP_HIGH = 1.0 + ADJUSTMENT_CAP
FACTOR = {"OUT": CLAMP_LOW, "DOUBT": 0.85, "BOOST": 1.0 + ADJUSTMENT_CAP / 2}

UPDATE_PREDICTION = """
UPDATE predictions
   SET adjusted_points   = :adjusted_points,
       adjustment_factor = :adjustment_factor,
       adjustment_reason = :adjustment_reason,
       news_url          = :news_url
 WHERE player_id = :player_id AND season = :season AND gameweek = :gameweek
"""


def _surname(second_name: str, web_name: str) -> str:
    for candidate in (second_name, web_name):
        parts = [p for p in (candidate or "").replace(".", " ").split() if len(p) >= 4]
        if parts:
            return parts[-1].lower()
    return (web_name or "").lower()


def classify(chunks: list[dict], surname: str) -> tuple[str, str, str] | None:
    """First (category, matched_phrase, url) whose chunk names the player, else None."""
    for chunk in chunks:
        body = chunk["text"].lower()
        if surname and surname not in body:
            continue
        for category, terms in CATEGORIES:
            for term in terms:
                if term in body:
                    return category, term.strip(), chunk["url"]
    return None


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT pr.player_id, pr.gameweek, pr.raw_points,
                   p.web_name, p.second_name
            FROM predictions pr
            JOIN players p ON p.player_id = pr.player_id
            WHERE pr.season = ? AND pr.raw_points IS NOT NULL
            """,
            (SEASON,),
        ).fetchall()

    try:
        from rag.retrieve import retrieve
    except Exception as error:  # noqa: BLE001
        retrieve = None
        print(f"retrieve unavailable ({error}); adjusted = raw for all", file=sys.stderr)

    updates: list[dict] = []
    adjusted_count = 0
    for row in rows:
        raw = row["raw_points"]
        factor, reason, url = 1.0, None, None

        if retrieve is not None:
            surname = _surname(row["second_name"], row["web_name"])
            try:
                chunks = retrieve(f"{row['web_name']} {row['second_name'] or ''}".strip())
            except Exception as error:  # noqa: BLE001
                chunks = []
                print(f"  retrieve failed for {row['web_name']}: {error}", file=sys.stderr)
            hit = classify(chunks, surname)
            if hit:
                category, phrase, url = hit
                factor = min(CLAMP_HIGH, max(CLAMP_LOW, FACTOR[category]))
                reason = f"{category}: matched '{phrase}' in recent news"
                adjusted_count += 1

        updates.append(
            {
                "player_id": row["player_id"],
                "season": SEASON,
                "gameweek": row["gameweek"],
                "adjusted_points": round(raw * factor, 2),
                "adjustment_factor": round(factor, 4),
                "adjustment_reason": reason,
                "news_url": url,
            }
        )

    with get_connection() as conn:
        conn.executemany(UPDATE_PREDICTION, updates)

    print(
        f"Adjusted {adjusted_count} of {len(updates)} predictions from news signals "
        f"(factor clamped to [{CLAMP_LOW:.2f}, {CLAMP_HIGH:.2f}])"
    )


if __name__ == "__main__":
    main()
