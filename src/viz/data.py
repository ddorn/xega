"""
Data utilities for XEGA visualizer.

File structures:
- Per-game result JSON (example fields):
  {
    "game": {
      "game": {"name": str, "code": str, ...},
      "players": [{"name": str, "id": str, "player_type": str, "options": {"model": str, "provider": str}}],
      "map_seed": str,
      "judge_model": str,
      ...
    },
    "game_results": [
      {"scores": {"black": float}, "xrt_history": [events...], "token_usage": {"black": {"input_tokens": int, "output_tokens": int}}},
      ...
    ],
    "scores": {"black": float},            # overall best/summary (optional)
    "token_usage": {"black": {...}}        # overall usage (per file, optional)
  }

- Benchmark JSON:
  {
    "benchmark": {"games": [...] , "benchmark_id": str},
    "game_results": [   # many entries
      {
        "game": {   # same structure as per-game header
          "game": {"name": str, "code": str, ...},
          "players": [...],
          "map_seed": str,
          "judge_model": str,
          ...
        },
        "game_results": [  # iterations (runs) for this game+seed+model
          {"scores": {"black": float}, "xrt_history": [events...]},
          ...
        ]
      },
      ...
    ]
  }

Event structure (within xrt_history):
- "elicit_request": includes registers (prompt context)
- "elicit_response": includes response text and token_usage
- "reward": includes per-token values under value.pairs when value.__TokenXentList__ is true
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

# ---------- File IO ----------


def find_result_files(base_dir: Path) -> list[Path]:
    results_dir = base_dir / "results"
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("*.json"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Benchmark helpers ----------


def is_benchmark_json(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    # Per-game files have a top-level 'game' dict. Benchmarks generally do not.
    if isinstance(data.get("game"), dict):
        return False
    entries = data.get("game_results")
    if not isinstance(entries, list) or not entries:
        return False
    first = entries[0]
    # Benchmark entries have nested 'game' dict and their own 'game_results' list
    return (
        isinstance(first, dict)
        and isinstance(first.get("game"), dict)
        and isinstance(first.get("game_results"), list)
    )


def select_benchmark_entry(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, str]:
    """Return (game_meta, player0, game_results, game_name, seed) via cascading selectors.
    Order: Game → Seed → Model.
    """
    entries = data.get("game_results") or []
    # Build lookup: game -> seed -> list[(model, idx)]
    by_game: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for i, entry in enumerate(entries):
        g = (entry.get("game") or {}).get("game") or {}
        name = g.get("name", "-")
        seed = (entry.get("game") or {}).get("map_seed", "-")
        players = (entry.get("game") or {}).get("players") or []
        model = (players[0].get("options") or {}).get("model", "-") if players else "-"
        by_game.setdefault(name, {}).setdefault(seed, []).append((model, i))

    # Sidebar cascading selectors
    game_name = st.sidebar.selectbox("Game", options=sorted(by_game.keys()))
    seeds = sorted(by_game.get(game_name, {}).keys())
    seed = st.sidebar.selectbox("Seed", options=seeds)
    models = [m for m, _ in sorted(by_game[game_name][seed], key=lambda t: t[0])]
    model = st.sidebar.selectbox("Model", options=models)

    # Resolve selected entry index
    idx = next(i for m, i in by_game[game_name][seed] if m == model)
    entry = entries[idx]

    game_meta = (entry.get("game") or {}).get("game") or {}
    players = (entry.get("game") or {}).get("players") or []
    player0 = players[0] if players else {}
    game_results = entry.get("game_results") or []
    return game_meta, player0, game_results, game_name, seed


def collect_moves_across_models(
    data: dict[str, Any], target_game: str, target_seed: str
) -> list[dict[str, Any]]:
    """Collect moves across all models for the given game+seed.
    Returns list of {model, run, score, move}.
    """
    rows: list[dict[str, Any]] = []
    for entry in data.get("game_results", []):
        g = (entry.get("game") or {}).get("game") or {}
        name = g.get("name")
        seed = (entry.get("game") or {}).get("map_seed")
        if name != target_game or seed != target_seed:
            continue
        players = (entry.get("game") or {}).get("players") or []
        model = (players[0].get("options") or {}).get("model", "-") if players else "-"
        for idx, run in enumerate(entry.get("game_results") or []):
            move_text = ""
            for ev in run.get("xrt_history", []):
                if ev.get("type") == "elicit_response":
                    move_text = ev.get("response") or ""
                    break
            score = (run.get("scores") or {}).get("black")
            rows.append({"model": model, "run": idx, "score": score, "move": move_text})
    return rows


def collect_moves_single_file(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Collect moves across runs for a single per-game file.
    Returns (model_name, rows[{model, run, score, move}]).
    """
    players = (data.get("game") or {}).get("players") or []
    model = (players[0].get("options") or {}).get("model", "-") if players else "-"
    rows: list[dict[str, Any]] = []
    for idx, run in enumerate(data.get("game_results") or []):
        move_text = ""
        for ev in run.get("xrt_history", []):
            if ev.get("type") == "elicit_response":
                move_text = ev.get("response") or ""
                break
        score = (run.get("scores") or {}).get("black")
        rows.append({"model": model, "run": idx, "score": score, "move": move_text})
    return model, rows


# ---------- Iteration helpers ----------


def collect_iterative_data(
    game_results: list[dict[str, Any]],
) -> tuple[list[str], list[list[float]]]:
    """Return (tokens, list_of_y_values_per_run) based on first reward in each run.
    Uses the token index order; lengths may differ across runs and will be clipped to the baseline.
    """
    baseline_tokens: list[str] = []
    y_per_run: list[list[float]] = []
    for run in game_results:
        pairs = None
        for ev in run.get("xrt_history", []):
            if ev.get("type") == "reward":
                val = ev.get("value") or {}
                if val.get("__TokenXentList__"):
                    pairs = val.get("pairs", [])
                    break
        if pairs is None:
            continue
        vals = [p[1] for p in pairs if isinstance(p, list) and len(p) == 2]
        if not baseline_tokens:
            baseline_tokens = [
                p[0] for p in pairs if isinstance(p, list) and len(p) == 2
            ]
        else:
            vals = vals[: len(baseline_tokens)]
            y_per_run.append(vals)
            continue
        y_per_run.append(vals)
    if baseline_tokens:
        L = len(baseline_tokens)
        y_per_run = [ys[:L] for ys in y_per_run]
    return baseline_tokens, y_per_run


# ---------- Embeddings (OpenAI) ----------


@st.cache_resource
def get_openai_client():
    from openai import OpenAI

    return OpenAI()


@st.cache_data(show_spinner=False)
def embed_texts_openai(
    texts: tuple[str, ...], model: str = "text-embedding-3-large"
) -> list[list[float]]:
    client = get_openai_client()
    inputs = list(texts)
    if not inputs:
        return []
    # Batch to avoid oversized 'input' payloads
    max_batch = 1000
    all_embeddings: list[list[float]] = []
    for i in range(0, len(inputs), max_batch):
        chunk = inputs[i : i + max_batch]
        # Replace empty strings with " " to avoid empty inputs
        chunk = [s or " " for s in chunk]
        resp = client.embeddings.create(model=model, input=chunk)
        # Preserve order
        all_embeddings.extend([item.embedding for item in resp.data])
    return all_embeddings


# ---------- UMAP ----------


def umap_reduce(
    embeddings: list[list[float]],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> list[tuple[float, float]]:
    import numpy as np
    from umap import UMAP

    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    xy = umap.fit_transform(np.asarray(embeddings, dtype=float))
    return [(float(x), float(y)) for x, y in xy]
