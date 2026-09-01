"""Turn the optimised squad into a plain-English writeup with one Claude call.

The prompt gets the XI, bench, captain/vice, each starter's predicted points and
any news flag from the RAG step. Claude returns a small JSON object which we
render to markdown. If ANTHROPIC_API_KEY is not set (e.g. a local run without
the secret) we fall back to a deterministic template so the pipeline still
completes.

Writes: table ``explanations``, ``site/data/explanation.json`` + ``.md``.

Run (after squad_optimizer.py):
    python src/explain/generate_explanation.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import EXPLANATION_MODEL, SEASON  # noqa: E402
from db import get_connection  # noqa: E402

SITE_DATA = Path(__file__).resolve().parent.parent.parent / "site" / "data"
SQUAD_JSON = SITE_DATA / "squad.json"

SYSTEM_PROMPT = (
    "You are an FPL analyst. You are given a squad that a points model and an "
    "optimiser produced. Explain it for a reader who knows football but not the "
    "model. Be concrete: cite predicted points and any injury/rotation news "
    "flag that is provided. Do not invent facts, stats, or news beyond what you "
    "are given. Respond with ONLY a JSON object, no prose around it, shaped:\n"
    '{"summary": "<2-3 sentences on the squad shape and overall approach>",\n'
    ' "captain_rationale": "<1-2 sentences on the captain and vice pick>",\n'
    ' "standout_picks": [{"player": "<name>", "reason": "<1-2 sentences>"}]}\n'
    "Include 2-4 standout picks."
)


def build_context() -> dict:
    squad = json.loads(SQUAD_JSON.read_text())
    gameweek = squad["gameweek"]

    with get_connection() as conn:
        news = {
            r["player_id"]: {"reason": r["adjustment_reason"], "url": r["news_url"]}
            for r in conn.execute(
                """
                SELECT player_id, adjustment_reason, news_url FROM predictions
                WHERE season = ? AND gameweek = ? AND adjustment_reason IS NOT NULL
                """,
                (SEASON, gameweek),
            )
        }

    def annotate(player: dict) -> dict:
        item = {
            "name": player["web_name"],
            "position": player["position"],
            "club": player["team"],
            "price": player["price"],
            "predicted_points": player["predicted_points"],
        }
        if player["player_id"] in news:
            item["news_flag"] = news[player["player_id"]]["reason"]
        return item

    return {
        "season": SEASON,
        "gameweek": gameweek,
        "formation": squad["formation"],
        "total_cost": squad["total_cost"],
        "projected_points": squad["predicted_points"],
        "captain": squad["captain"]["web_name"],
        "vice": squad["vice"]["web_name"],
        "starting_xi": [annotate(p) for p in squad["xi"]],
        "bench": [annotate(p) for p in squad["bench"]],
    }


def _call_claude(context: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    message = client.messages.create(
        model=EXPLANATION_MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(context, indent=2)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text)


def _fallback(context: dict) -> dict:
    top = sorted(
        context["starting_xi"], key=lambda p: p["predicted_points"], reverse=True
    )[:3]
    flagged = [p for p in context["starting_xi"] + context["bench"] if "news_flag" in p]
    summary = (
        f"A {context['formation']} squad costing GBP{context['total_cost']}m, "
        f"projected {context['projected_points']} points for GW{context['gameweek']}. "
        f"Built by maximising model-predicted points under FPL's budget, "
        f"formation and 3-per-club rules."
    )
    if flagged:
        summary += " News flags applied to: " + ", ".join(
            f"{p['name']} ({p['news_flag']})" for p in flagged
        ) + "."
    return {
        "summary": summary,
        "captain_rationale": (
            f"{context['captain']} is the highest projected starter, with "
            f"{context['vice']} as vice."
        ),
        "standout_picks": [
            {
                "player": p["name"],
                "reason": (
                    f"{p['club']} {p['position']} at GBP{p['price']}m, "
                    f"projected {p['predicted_points']} pts"
                    + (f"; note: {p['news_flag']}" if "news_flag" in p else "")
                ),
            }
            for p in top
        ],
    }


def render_markdown(context: dict, body: dict) -> str:
    lines = [
        f"# Suggested squad - {context['season']} GW{context['gameweek']}",
        "",
        body.get("summary", ""),
        "",
        f"**Formation** {context['formation']} | "
        f"**Cost** GBP{context['total_cost']}m | "
        f"**Projected** {context['projected_points']} pts",
        "",
        f"**Captain:** {context['captain']} | **Vice:** {context['vice']}  ",
        body.get("captain_rationale", ""),
        "",
        "## Standout picks",
        "",
    ]
    for pick in body.get("standout_picks", []):
        lines.append(f"- **{pick.get('player', '?')}** - {pick.get('reason', '')}")
    return "\n".join(lines) + "\n"


UPSERT_EXPLANATION = """
INSERT INTO explanations (season, gameweek, generated_at, markdown, json)
VALUES (:season, :gameweek, :generated_at, :markdown, :json)
ON CONFLICT(season, gameweek) DO UPDATE SET
    generated_at = excluded.generated_at,
    markdown     = excluded.markdown,
    json         = excluded.json
"""


def main() -> None:
    context = build_context()

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            body = _call_claude(context)
            source = EXPLANATION_MODEL
        except Exception as error:  # noqa: BLE001
            print(f"Claude call failed ({error}); using template", file=sys.stderr)
            body, source = _fallback(context), "template (api-error)"
    else:
        print("ANTHROPIC_API_KEY unset; using deterministic template", file=sys.stderr)
        body, source = _fallback(context), "template (no-key)"

    body.setdefault("standout_picks", [])
    markdown = render_markdown(context, body)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "season": context["season"],
        "gameweek": context["gameweek"],
        "generated_at": generated_at,
        "source": source,
        **body,
    }

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "explanation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (SITE_DATA / "explanation.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    with get_connection() as conn:
        conn.execute(
            UPSERT_EXPLANATION,
            {
                "season": context["season"],
                "gameweek": context["gameweek"],
                "generated_at": generated_at,
                "markdown": markdown,
                "json": json.dumps(payload),
            },
        )

    print(f"Explanation written ({source}); {len(body['standout_picks'])} standout picks")


if __name__ == "__main__":
    main()
