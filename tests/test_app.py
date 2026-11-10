"""Smoke tests for the Streamlit front end, run headlessly with Streamlit's AppTest."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from moonsim.presets import PRESETS  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def fresh():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_first_run_shows_a_story():
    at = fresh()
    assert not at.exception
    assert at.title[0].value == "2thamoon"
    assert at.subheader, "the headline for the default scenario should render"


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_each_preset_deep_links_and_runs(preset):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["scenario"] = preset.key
    at.run()
    assert not at.exception
    assert preset.blurb.replace("$", "\\$") in [m.value for m in at.markdown]
    assert at.session_state["preset_pick"] == preset.key


def test_deep_mode_reveals_every_knob():
    at = fresh()
    shallow = len(at.slider) + len(at.number_input) + len(at.toggle) + len(at.selectbox)
    at.toggle(key="deep").set_value(True).run()
    deep = len(at.slider) + len(at.number_input) + len(at.toggle) + len(at.selectbox)
    assert not at.exception
    assert deep > shallow + 50


def test_changing_a_knob_reruns_the_sim():
    at = fresh()
    before = at.subheader[0].value
    at.number_input(key="k:run.seed").set_value(12345).run()
    assert not at.exception
    assert at.subheader[0].value != before


def test_manual_attack_orders_and_mode_switch():
    at = fresh()
    at.selectbox(key="k:attacker.style").set_value("manual").run()
    assert not at.exception
    assert at.session_state["flat"]["attacker.style"] == "manual"
    at.selectbox(key="k:run.mode").set_value("stablecoin").run()
    assert not at.exception
    assert "peg" in at.subheader[0].value


def test_luck_check_runs():
    at = fresh()
    at.toggle(key="luck_on").set_value(True).run()
    assert not at.exception
    assert any("seeds end above" in c.value for c in at.caption)
