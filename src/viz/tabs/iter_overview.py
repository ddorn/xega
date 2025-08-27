from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from viz.data import collect_iterative_data


def render(game_results: list[dict[str, Any]]) -> None:
    st.subheader("Player moves across runs")
    moves_rows: list[dict[str, Any]] = []
    scores_over_runs: list[float] = []
    move_texts: list[str] = []

    for idx, run in enumerate(game_results):
        move_text = None
        in_tok = None
        for ev in run.get("xrt_history", []):
            if ev.get("type") == "elicit_response":
                move_text = ev.get("response")
                usage = ev.get("token_usage") or {}
                in_tok = usage.get("input_tokens")
                break
        score = (run.get("scores") or {}).get("black")
        if isinstance(score, int | float):
            scores_over_runs.append(score)
            move_texts.append(move_text or "")
        moves_rows.append(
            {
                "run": idx,
                "score": score,
                "in_tokens": in_tok,
                "move": move_text or "",
            }
        )

    st.dataframe(moves_rows, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Score across runs")
    if scores_over_runs:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores_over_runs))),
                y=scores_over_runs,
                mode="lines+markers",
                marker=dict(size=6),
                hovertemplate="Run %{x}<br>Score %{y:.3f}<br>Move: %{customdata}",
                customdata=[[m] for m in move_texts],
                name="score",
            )
        )
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Run",
            yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scores available to plot.")

    st.divider()
    st.subheader("Per-token reward across runs")
    tokens, y_per_run = collect_iterative_data(game_results)
    if not tokens or not y_per_run:
        st.info("No token-level reward data found across runs.")
        return

    from plotly.colors import sample_colorscale

    colors = sample_colorscale(
        "Viridis", [i / max(1, len(y_per_run) - 1) for i in range(len(y_per_run))]
    )
    fig = go.Figure()
    indices = list(range(len(tokens)))
    for i, ys in enumerate(y_per_run):
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=ys,
                mode="lines",
                line=dict(color=colors[i], width=2),
                name=f"Run {i}",
                customdata=tokens,
                hovertemplate="%{customdata}: %{y:.3f}",
            )
        )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Tokens in judge text",
        yaxis_title="Reward value (per token)",
        xaxis=dict(
            tickmode="array",
            tickvals=indices,
            ticktext=tokens,
            tickangle=90,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
