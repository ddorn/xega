from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_run(run: dict[str, Any], provider: str | None, model: str | None) -> None:
    scores = run.get("scores", {})
    run_score = scores.get("black")
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Run score (black)",
            f"{run_score:.3f}" if isinstance(run_score, int | float) else "-",
        )
    with cols[1]:
        st.metric("Provider", provider or "-")
    with cols[2]:
        st.metric("Model", model or "-")

    token_usage = run.get("token_usage", {}).get("black") or {}
    if token_usage:
        st.caption(
            f"Tokens — input: {token_usage.get('input_tokens', '-')} • output: {token_usage.get('output_tokens', '-')}"
        )

    st.divider()

    events: list[dict[str, Any]] = run.get("xrt_history", [])
    if not events:
        st.info("No history events found in this run.")
        return

    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "reveal":
            with st.expander(
                f"Reveal initial state (line {ev.get('line_num')})", expanded=False
            ):
                values = ev.get("values", {})
                for k, v in values.items():
                    st.write(f"{k} =")
                    st.code(str(v))
        elif ev_type == "elicit_request":
            with st.container(border=True):
                st.markdown("**API request**")
                left, right = st.columns(2)
                with left:
                    st.write(f"Player: {ev.get('player', '-')}")
                    st.write(
                        f"Var: `{ev.get('var_name', '-')}` • Max len: {ev.get('max_len', '-')}"
                    )
                with right:
                    st.write(f"Line: {ev.get('line', '-')}")
                    st.write(f"Line num: {ev.get('line_num', '-')}")
                with st.expander("Registers (prompt context)"):
                    registers = ev.get("registers", {})
                    if registers:
                        non_empty = {
                            k: v
                            for k, v in registers.items()
                            if isinstance(v, str) and v.strip()
                        }
                        if not non_empty:
                            st.write("All registers empty.")
                        else:
                            for k, v in non_empty.items():
                                st.write(f"`{k}`:")
                                st.code(v)
        elif ev_type == "elicit_response":
            with st.container(border=True):
                st.markdown("**API response / Player move**")
                st.write(f"Player: {ev.get('player', '-')}")
                st.write(f"Line: {ev.get('line', '-')} (#{ev.get('line_num', '-')})")
                response = ev.get("response", "")
                st.text_area("Response", response, height=100)
                usage = ev.get("token_usage", {})
                if usage:
                    st.caption(
                        f"Tokens — input: {usage.get('input_tokens', '-')} • output: {usage.get('output_tokens', '-')}"
                    )
        elif ev_type == "reward":
            val = ev.get("value", {})
            if val and val.get("__TokenXentList__"):
                pairs = val.get("pairs", [])
                tokens = [p[0] for p in pairs if isinstance(p, list) and len(p) == 2]
                values = [p[1] for p in pairs if isinstance(p, list) and len(p) == 2]
                if values:
                    avg = sum(values) / len(values)
                    st.markdown(
                        f"**Reward (line {ev.get('line_num', '-')})** — per-token entries: {len(values)} • avg: {avg:.3f}"
                    )
                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=list(range(len(tokens))),
                                y=values,
                                mode="lines+markers",
                                hovertemplate="%{customdata}: %{y:.3f}",
                                customdata=tokens,
                            )
                        ],
                        layout=go.Layout(
                            height=220, margin=dict(l=10, r=10, t=10, b=10)
                        ),
                    )
                    fig.update_xaxes(
                        tickmode="array",
                        tickvals=list(range(len(tokens))),
                        ticktext=tokens,
                        tickangle=90,
                        title_text="Tokens in judge text",
                    )
                    fig.update_yaxes(title_text="Reward value (per token)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No token-level reward details available.")
        else:
            with st.expander(f"Event: {ev_type}"):
                st.json(ev)


def render(
    game_results: list[dict[str, Any]], provider: str | None, model: str | None
) -> None:
    run_idx = st.slider(
        "Run index", min_value=0, max_value=len(game_results) - 1, value=0
    )
    st.subheader(f"Run #{run_idx}")
    run = game_results[run_idx]
    render_run(run, provider=provider, model=model)
