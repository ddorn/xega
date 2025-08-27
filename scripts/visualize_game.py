"""
XEGA Streamlit Visualizer

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
    "scores": {"black": float},            # overall best/summary
    "token_usage": {"black": {...}}        # overall usage (per file)
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

from viz.app import main

if __name__ == "__main__":
    main()
