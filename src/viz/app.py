from __future__ import annotations

from pathlib import Path

import streamlit as st

from viz.data import (
    find_result_files,
    is_benchmark_json,
    load_json,
    select_benchmark_entry,
)
from viz.tabs import exploration as tab_explore
from viz.tabs import iter_overview as tab_iter
from viz.tabs import move_map as tab_map
from viz.tabs import run_details as tab_details


def main() -> None:
    st.set_page_config(page_title="XEGA Game Visualizer", layout="wide")
    st.title("XEGA Game Visualizer")

    base_dir = Path(__file__).resolve().parents[2]

    # Sidebar: pick a results file
    st.sidebar.header("Select Results File")
    all_files = find_result_files(base_dir)
    labels = [str(p.relative_to(base_dir)) for p in all_files]
    selected_label = st.sidebar.selectbox("Result JSON", options=labels or [""])

    if not labels:
        st.warning("No results JSON files found under `results/`. Add one to view.")
        st.stop()

    selected_path = base_dir / selected_label
    data = load_json(selected_path)

    is_bench = is_benchmark_json(data)
    if is_bench:
        st.info("Benchmark file detected: pick Game → Seed → Model.")
        game_meta, player0, game_results, game_name, seed = select_benchmark_entry(data)
    else:
        game_meta = (data.get("game") or {}).get("game") or {}
        players = (data.get("game") or {}).get("players") or []
        player0 = players[0] if players else {}
        game_results = data.get("game_results", [])
        game_name = game_meta.get("name", "-")
        seed = (data.get("game") or {}).get("map_seed", "-")

    provider = (player0.get("options") or {}).get("provider")
    model = (player0.get("options") or {}).get("model")

    # Collapsed metadata
    with st.expander("Game metadata", expanded=False):
        st.write("Name:", game_meta.get("name", "-"))
        st.write("Prepare function:")
        st.code(game_meta.get("code", ""))

    st.subheader(f"Game: {game_meta.get('name', '-')}")
    if not is_bench:
        st.caption(f"Judge model: {data.get('game', {}).get('judge_model', '-')}")

    if not game_results:
        st.info("No game results found.")
        st.stop()

    # Segmented control for views
    options = ["Overview", "Run details", "Move map"] + (
        ["Exploration"] if is_bench else []
    )
    selection = st.segmented_control(
        "View", options, selection_mode="single", default="Overview"
    )

    if selection == "Overview":
        tab_iter.render(game_results)
    elif selection == "Run details":
        tab_details.render(game_results, provider=provider, model=model)
    elif selection == "Move map":
        # Always available; for per-game, the renderer will aggregate runs for that model
        if is_bench:
            tab_map.render(data, game_name=game_name, seed=seed)
        else:
            tab_map.render(data)
    elif selection == "Exploration" and is_bench:
        tab_explore.render(data, game_name=game_name)


if __name__ == "__main__":
    main()
