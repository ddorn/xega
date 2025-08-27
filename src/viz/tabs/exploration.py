from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from viz.data import embed_texts_openai, is_benchmark_json


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(na, nb))


def _get_game_models_entries(
    data: dict[str, Any], game_name: str
) -> dict[str, list[dict[str, Any]]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in data.get("game_results", []):
        g = (entry.get("game") or {}).get("game") or {}
        name = g.get("name")
        if name != game_name:
            continue
        players = (entry.get("game") or {}).get("players") or []
        model = (
            str((players[0].get("options") or {}).get("model") or "-")
            if players
            else "-"
        )
        by_model.setdefault(model, []).append(entry)
    return by_model


def _collect_seed_embeddings(
    entry: dict[str, Any],
) -> tuple[list[str], np.ndarray, list[float], list[int]]:
    # Returns (moves_texts, embeddings ndarray [T, D], scores list[T], run_indices list[T]) for one seed/model entry
    runs = entry.get("game_results") or []
    moves: list[str] = []
    scores: list[float] = []
    run_idx: list[int] = []
    for i, run in enumerate(runs):
        move_text = ""
        for ev in run.get("xrt_history", []):
            if ev.get("type") == "elicit_response":
                move_text = ev.get("response") or ""
                break
        score = (run.get("scores") or {}).get("black")
        moves.append(move_text)
        scores.append(score if isinstance(score, int | float) else float("nan"))
        run_idx.append(i)
    # Embed all moves together to leverage cache
    emb_list = embed_texts_openai(tuple(moves))
    embs = np.asarray(emb_list, dtype=float)
    return moves, embs, scores, run_idx


def render(data: dict[str, Any], game_name: str) -> None:
    if not is_benchmark_json(data):
        st.info(
            "Exploration view currently uses multiple seeds per game (benchmark files). Select a benchmark file."
        )
        return

    by_model = _get_game_models_entries(data, game_name)
    if not by_model:
        st.info("No entries for this game.")
        return

    all_models = sorted(by_model.keys())
    default_models = all_models[:5]
    selected_models: list[str] = st.multiselect(
        "Models (for distributions)", options=all_models, default=default_models
    )
    if not selected_models:
        st.warning("Select at least one model.")
        return

    # 1) Centroid spread distribution per model (across seeds)
    # For each seed (entry) and model, compute mean distance of runs to that seed centroid
    dist_rows: list[dict[str, Any]] = []
    for model in selected_models:
        for entry in by_model.get(model, []):
            moves, embs, scores, run_idx = _collect_seed_embeddings(entry)
            if embs.size == 0:
                continue
            centroid = embs.mean(axis=0)
            dists = np.array([_cosine_distance(centroid, e) for e in embs])
            radius = float(dists.mean())
            seed_val = (entry.get("game") or {}).get("map_seed")
            dist_rows.append({"model": model, "seed": seed_val, "radius": radius})

    if dist_rows:
        df = pd.DataFrame(dist_rows)
        st.subheader("Spread around centroid (mean distance per seed)")
        fig = px.box(
            df,
            x="model",
            y="radius",
            points="all",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Light24,
        )
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Mean cosine distance to seed centroid",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to compute centroid spread.")

    st.divider()

    # 2) Drift from start vs. score scatter for a chosen model (across seeds)
    scatter_model = st.selectbox(
        "Model for drift vs score", options=all_models, index=0
    )
    pts: list[dict[str, Any]] = []
    for entry in by_model.get(scatter_model, []):
        moves, embs, scores, run_idx = _collect_seed_embeddings(entry)
        if embs.shape[0] == 0:
            continue
        start = embs[0]
        seed_val = (entry.get("game") or {}).get("map_seed")
        for i in range(embs.shape[0]):
            drift = _cosine_distance(start, embs[i])
            pts.append(
                {
                    "seed": str(seed_val),
                    "run": run_idx[i],
                    "drift": drift,
                    "score": scores[i],
                    "move": moves[i],
                }
            )

    if pts:
        df2 = pd.DataFrame(pts)
        st.subheader("Drift from start vs. score")
        fig2 = px.scatter(
            df2,
            x="drift",
            y="score",
            color="seed",
            hover_data={"run": True, "move": True, "seed": True},
            color_discrete_sequence=px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Light24,
        )
        fig2.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Cosine distance from first move",
            yaxis_title="Score",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No points for drift vs score.")
