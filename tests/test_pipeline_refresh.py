import json
import sys
from types import SimpleNamespace

import pytest

import pipeline
from db import get_connection, init_db
from models import predict


def test_cached_historical_rows_trigger_identity_backfill(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, web_name, position, team) VALUES (1, 'A', 'MID', 'Now')")
        conn.execute("INSERT INTO player_gameweek_stats (player_id, season, gameweek, total_points) "
                     "VALUES (1, '2021-22', 1, 2)")
    monkeypatch.setattr(pipeline, "get_connection", lambda: get_connection(path))
    assert pipeline.seasons_needing_identity() == ["2021-22"]


def test_old_models_are_retrained_when_features_change(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for pos in ("GK", "DEF", "MID", "FWD"):
        (models / f"{pos}.joblib").write_bytes(b"old")
    (models / "manifest.json").write_text(json.dumps({"trained_at": "2099-01-01T00:00:00+00:00",
                                                       "feature_columns": ["old_feature"]}))
    monkeypatch.setattr(pipeline, "MODELS_DIR", models)
    calls = []
    monkeypatch.setitem(sys.modules, "models.train", SimpleNamespace(production_fit=lambda: calls.append("retrained")))
    monkeypatch.setattr(pipeline, "_RETRAINED", False)
    pipeline.stage_ensure_models()
    assert calls == ["retrained"]


def test_predict_rejects_models_with_old_feature_schema(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text(json.dumps({"feature_columns": ["old_feature"]}))
    monkeypatch.setattr(predict, "MODELS_DIR", models)
    with pytest.raises(SystemExit, match="retrain"):
        predict.load_models()
