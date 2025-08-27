from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from viz.data import (
    collect_moves_across_models,
    collect_moves_single_file,
    embed_texts_openai,
    is_benchmark_json,
    umap_reduce,
)


def render(
    data: dict[str, Any], game_name: str | None = None, seed: str | None = None
) -> None:
    st.subheader("Move map")
    st.caption(
        "UMAP projection of move embeddings. Color = model. Hover shows run, score, move."
    )

    rows: list[dict[str, Any]]
    if is_benchmark_json(data) and game_name is not None and seed is not None:
        rows = collect_moves_across_models(
            data, target_game=game_name, target_seed=seed
        )
    else:
        model, rows_single = collect_moves_single_file(data)
        rows = rows_single

    if not rows:
        st.info("No moves found for this selection.")
        return

    moves = tuple(r["move"] or "" for r in rows)
    with st.spinner("Embedding moves..."):
        embs = embed_texts_openai(moves)
    xy = umap_reduce(embs)

    df = pd.DataFrame(
        {
            "x": [p[0] for p in xy],
            "y": [p[1] for p in xy],
            "model": [r.get("model", "-") for r in rows],
            "run": [r.get("run") for r in rows],
            "score": [r.get("score") for r in rows],
            "move": [r.get("move") for r in rows],
        }
    )

    # Expanded color palette: qualitative + cyclic fallback
    palette = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="model",
        color_discrete_sequence=palette,
        hover_data={"run": True, "score": ":.3f", "move": True, "x": False, "y": False},
        opacity=0.9,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
