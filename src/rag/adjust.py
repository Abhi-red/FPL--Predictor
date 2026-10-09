"""Nudge raw predictions using retrieved news - only on a clear, specific signal.

For every player with a raw prediction for the upcoming gameweek we retrieve
recent news. A chunk only counts if an identifying name and a bounded
availability phrase occur in the same sentence or clause:

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
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import ADJUSTMENT_CAP, SEASON  # noqa: E402
from db import get_connection  # noqa: E402

OUT_TERMS = (
    "ruled out", "sidelined", "will miss", "out for", "suspended",
    "banned", "surgery", "long-term injury", "long term injury",
    "not in the squad", "won't play", "will not play", "months out", "acl injury", "acl tear",
)
DOUBT_TERMS = (
    "doubt", "knock", "late test", "late fitness", "rotation risk",
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


def _player_names(first_name: str, second_name: str, web_name: str) -> list[str]:
    """Use identifying names; a shared surname alone is not enough evidence."""
    first = (first_name or "").strip()
    second = (second_name or "").strip()
    names = [f"{first} {second}".strip(), f"{first} {second.split()[-1]}" if second else ""]
    if len((web_name or "").split()) > 1:
        names.append(web_name)
    return [name for name in dict.fromkeys(names) if len(name.split()) > 1]


def classify(chunks: list[dict], player_names: str | list[str]) -> tuple[str, str, str] | None:
    """Return a signal only when a named player and claim share a sentence."""
    names = [player_names] if isinstance(player_names, str) else player_names
    patterns = [re.compile(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)") for name in names]
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?;])\s+|,\s*|\b(?:but|while|and|whereas)\b", chunk["text"].lower()):
            if not any(pattern.search(sentence) for pattern in patterns):
                continue
            for category, terms in CATEGORIES:
                for term in terms:
                    for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sentence):
                        before = sentence[max(0, match.start() - 60):match.start()]
                        if re.search(r"\b(?:no|not|never|without|cannot|can't|isn't|wasn't|hasn't|haven't|won't|wouldn't|couldn't|shouldn't)\b(?:\s+\w+){0,4}\s*$", before):
                            continue
                        if term == "out for" and re.search(r"\b(look|watch)\s+$", before):
                            continue
                        return category, term, chunk["url"]
    return None


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT pr.player_id, pr.gameweek, pr.raw_points,
                   p.web_name, p.first_name, p.second_name
            FROM predictions pr
            JOIN players p ON p.player_id = pr.player_id
            WHERE pr.season = ? AND pr.raw_points IS NOT NULL
              AND pr.gameweek = (SELECT MAX(gameweek) FROM predictions WHERE season = ?)
            """,
            (SEASON, SEASON),
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
            names = _player_names(row["first_name"], row["second_name"], row["web_name"])
            try:
                chunks = retrieve(f"{row['web_name']} {row['second_name'] or ''}".strip())
            except Exception as error:  # noqa: BLE001
                chunks = []
                print(f"  retrieve failed for {row['web_name']}: {error}", file=sys.stderr)
            hit = classify(chunks, names)
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
