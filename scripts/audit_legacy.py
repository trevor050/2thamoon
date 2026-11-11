"""Reproduce the numbers quoted in the README about the original (v1) engine.

The v1 app is loaded straight from git history (commit 2e4c96a, the last commit before the
rewrite) and executed in memory, so nothing is written to disk. Needs the app dependencies
(streamlit, pandas, numpy, plotly) because v1 imported streamlit at module level.

    python scripts/audit_legacy.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import types
import warnings

LEGACY_COMMIT = "2e4c96a"


def legacy_source() -> str:
    return subprocess.run(["git", "show", f"{LEGACY_COMMIT}:app.py"], check=True, capture_output=True, text=True).stdout


def dead_settings(src: str) -> list[str]:
    """Config fields that the v1 engine (everything before the UI code) never reads."""
    engine = src[src.index("def _build_attacker_states") : src.index("def render_sidebar_guide")]
    engine += src[src.index("def _annual_to_step_rate") : src.index("def render_simulation_tab(")]
    # Free text and bookkeeping are not settings; AlgorithmSettings.mode is applied by the UI's playbooks.
    not_knobs = {"objective_note", "engine_version", "goal_note", "mode"}
    dead = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name.endswith(("Settings", "Config", "Plan")):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    used = re.search(rf"\.{name}\b", engine) or re.search(rf"[\'\"]{name}[\'\"]", engine)
                    if not used and name not in not_knobs:
                        dead.append(f"{node.name}.{name}")
    return dead


def load_legacy():
    src = legacy_source()
    module = types.ModuleType("legacy_app")
    sys.modules["legacy_app"] = module  # dataclasses look their module up while being created
    warnings.filterwarnings("ignore")
    exec(compile(src, "legacy_app.py", "exec"), module.__dict__)  # noqa: S102 - our own historical source
    return module.__dict__


def main() -> None:
    v1 = load_legacy()
    out = v1["run_simulation"](v1["SimulationConfig"](), v1["PolicySettings"](), [], False, v1["AttackerSettings"]())
    tl = out.timeline
    peak = tl["market_cap"].idxmax()
    print(f"v1 default run, {len(tl)} steps")
    print(
        f"  peak market cap ${tl.market_cap[peak]:,.0f} at step {peak}, with {tl.users[peak]:,.0f} users holding "
        f"${tl.community_capital[peak]:,.0f} of total capital"
    )
    supply = tl.free_float_supply + tl.locked_supply + tl.treasury_tokens
    unexplained = (supply.diff() - tl.minted_step + tl.burned_step).fillna(0.0)
    print(f"  tokens that appear in total supply with no mint record: {unexplained.sum():,.0f}")

    atk = v1["AttackerSettings"](auto_enabled=True)
    out = v1["run_simulation"](v1["SimulationConfig"](), v1["PolicySettings"](), [], True, atk)
    st = out.attacker_state
    start = sum(p.capital for p in atk.profiles) + atk.capital
    print(f"  attacker (auto mode): starts with ${start:,.0f}, ends with ${st.attacker_cash.iloc[-1]:,.0f} cash")
    dead = dead_settings(legacy_source())
    print(f"  {len(dead)} settings the engine never reads: {', '.join(dead)}")


if __name__ == "__main__":
    main()
