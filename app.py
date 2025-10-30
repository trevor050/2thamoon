import math
import random
from collections import Counter
from dataclasses import dataclass, asdict, field, fields
import inspect
from itertools import count
from typing import List, Dict, Any, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PLOTLY_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
_plotly_key_counter = count()


@dataclass
class AttackEvent:
    name: str
    start_step: int
    duration: int
    side: str  # "buy", "sell", "liquidity", "mixed"
    magnitude: float  # as fraction of liquidity depth
    notes: str = ""
    liquidity_drop: float = 0.0  # fraction reduction in depth when active


@dataclass
class HypeSettings:
    enable_background_hype: bool = True
    random_hype_chance: float = 0.08  # annualized probability converted per step
    viral_spike_chance: float = 0.015
    min_duration_steps: int = 2
    max_duration_steps: int = 9
    base_intensity: float = 0.4
    viral_intensity_multiplier: float = 2.4
    whale_bias: float = 0.35
    quick_flip_ratio: float = 0.45
    holder_ratio: float = 0.22
    retention_decay: float = 0.35
    hype_cooldown_steps: int = 12


@dataclass
class MarketRegimeSettings:
    enabled: bool = True
    mean_duration_years: float = 2.4
    duration_jitter_years: float = 0.9
    expansion_bias: float = 0.55
    neutral_chance: float = 0.25
    expansion_growth_bonus: float = 0.004
    expansion_confidence_bonus: float = 0.06
    expansion_liquidity_bonus: float = 0.35
    contraction_growth_penalty: float = 0.005
    contraction_churn_bonus: float = 0.028
    contraction_confidence_hit: float = 0.07
    contraction_liquidity_hit: float = 0.3
    capital_cycle_swing: float = 0.3
    capacity_drift_speed: float = 0.012
    capacity_floor_shift: float = -0.3
    innovation_chance: float = 0.05  # annual probability of structural innovation
    innovation_capacity_boost: float = 0.12
    innovation_decay_rate: float = 0.35  # annual decay of innovation premium


@dataclass
class AlgorithmObjective:
    name: str = "Stability"
    metric: Literal["price", "peg_deviation", "growth", "volatility", "treasury_nav", "custom_index"] = "peg_deviation"
    target_value: float = 0.0
    tolerance: float = 0.05
    comparison: Literal["track", "at_most", "at_least", "maximize", "minimize"] = "track"
    weight: float = 1.0
    horizon_years: float = 0.5
    decay_rate: float = 0.4
    dynamic_trigger: Optional[float] = None
    enabled: bool = True


@dataclass
class AlgorithmModuleSettings:
    circuit_breaker: bool = True
    circuit_threshold: float = 0.07
    circuit_cooldown_steps: int = 4
    adaptive_mint: bool = True
    adaptive_burn: bool = True
    liquidity_support: bool = True
    alpha_stabilizer: bool = False
    peg_anchor: Literal["gold", "fundamental", "hybrid", "none"] = "hybrid"
    trend_follow: bool = False
    drawdown_guard: bool = True
    drawdown_threshold: float = 0.18
    floor_price: Optional[float] = None
    ceiling_price: Optional[float] = None
    velocity_targets: bool = False
    objective_lock_when_met: bool = False


@dataclass
class CrashDefenseSettings:
    enabled: bool = False
    detection_window: int = 5
    drop_threshold_pct: float = 12.0
    resource_commitment_pct: float = 40.0
    aggression_pct: float = 60.0
    gas_subsidy_share_pct: float = 25.0
    circuit_lock_steps: int = 6
    stabilize_steps: int = 6
    cooldown_steps: int = 12
    gas_efficiency: float = 0.35


@dataclass
class AlgorithmSettings:
    enabled: bool = False
    master_switch: bool = False
    mode: Literal["defend", "grow", "custom", "stabilize"] = "defend"
    objectives: List[AlgorithmObjective] = field(default_factory=lambda: [AlgorithmObjective()])
    modules: AlgorithmModuleSettings = field(default_factory=AlgorithmModuleSettings)
    crash_defense: CrashDefenseSettings = field(default_factory=CrashDefenseSettings)
    goal_note: str = ""
    discretionary_budget: float = 0.0
    treasury_ramp_years: float = 0.0
    custom_params: Dict[str, Any] = field(default_factory=dict)
    leverage_controls: Dict[str, bool] = field(default_factory=lambda: {
        "buy": True,
        "sell": True,
        "mint_adjust": True,
        "burn_adjust": True,
        "hype_boost": True,
    })


@dataclass
class TokenSupplyPlan:
    initial_circulating: float = 1_000_000.0
    initial_locked: float = 85_000_000.0
    initial_treasury: float = 14_000_000.0
    launch_float_release: float = 0.08
    goal_price: float = 2_000.0
    goal_release_multiplier: float = 4.0
    baseline_emission_per_user: float = 0.35
    goal_emission_per_user: float = 2.0
    halving_interval_steps: int = 36
    halving_factor: float = 0.5
    tx_tax_rate: float = 0.012
    burn_share_of_tax: float = 0.75
    burn_vault_share: float = 0.25
    burn_vault_release_threshold: float = 0.65
    burn_vault_release_fraction: float = 0.35
    burn_pause_price: float = 0.75
    burn_escalate_price: float = 1.8
    supply_hard_cap: float = 250_000_000.0
    unlock_slope: float = 0.015
    unlock_jitter: float = 0.25
    inflation_guard_price: float = 0.55
    inflation_guard_cooldown: int = 6
    supply_regime: Literal["balanced", "decay", "hard_cap", "adaptive"] = "balanced"
    decay_half_life_years: float = 3.5
    adaptive_floor_price: float = 0.8
    adaptive_ceiling_price: float = 2.8
    price_trigger_unlock: Optional[float] = None
    price_trigger_unlock_fraction: float = 0.15
    price_trigger_burn: Optional[float] = None
    price_trigger_burn_fraction: float = 0.02
    burn_floor_supply: float = 5_000_000.0
    supply_reversal_price: Optional[float] = None
    supply_reversal_fraction: float = 0.4
    absolute_supply_enabled: bool = False
    absolute_supply_tokens: Optional[float] = None


@dataclass
class AttackerProfile:
    name: str
    style: Literal["momentum", "pump_and_dump", "liquidity_sapper", "arb_sniper", "scripted"]
    capital: float
    aggression: float
    skill: float
    patience: float
    perception: float
    reinvest: bool = True
    leverage: float = 1.0


def _default_attacker_profiles() -> List[AttackerProfile]:
    return [
        AttackerProfile(
            name="Impulse Cartel",
            style="pump_and_dump",
            capital=1_500_000.0,
            aggression=0.55,
            skill=0.65,
            patience=0.6,
            perception=0.5,
            reinvest=True,
            leverage=1.4,
        ),
        AttackerProfile(
            name="Velocity Desk",
            style="momentum",
            capital=900_000.0,
            aggression=0.4,
            skill=0.58,
            patience=0.35,
            perception=0.62,
            reinvest=True,
            leverage=1.1,
        ),
        AttackerProfile(
            name="Depth Eaters",
            style="liquidity_sapper",
            capital=650_000.0,
            aggression=0.32,
            skill=0.5,
            patience=0.45,
            perception=0.4,
            reinvest=False,
            leverage=0.9,
        ),
        AttackerProfile(
            name="Spread Hounds",
            style="arb_sniper",
            capital=500_000.0,
            aggression=0.28,
            skill=0.72,
            patience=0.5,
            perception=0.75,
            reinvest=False,
            leverage=1.6,
        ),
    ]


@dataclass
class SimulationConfig:
    years: int = 5
    steps_per_year: int = 48
    initial_gold_price: float = 1900.0
    gold_drift_annual: float = 0.025
    gold_vol_annual: float = 0.06
    initial_token_price: float = 0.0
    initial_users: int = 250
    initial_free_float: float = 1_250_000.0
    founder_locked: float = 90_000_000.0
    initial_treasury_tokens: float = 8_000_000.0
    intrinsic_value_start: float = 0.25
    retail_wallet_mean: float = 2_800.0
    retail_wallet_spread: float = 0.35
    whale_user_fraction: float = 0.045
    whale_balance_multiplier: float = 22.0
    speculator_fraction: float = 0.12
    speculator_base_balance: float = 1_800.0
    capital_noise: float = 0.22
    intrinsic_growth_exponent: float = 0.35
    intrinsic_confidence_weight: float = 0.45
    intrinsic_liquidity_weight: float = 0.25
    intrinsic_noise: float = 0.065
    gold_guidance_strength: float = 0.0
    hype_start: float = 0.05
    hype_decay: float = 0.04
    hype_sensitivity_growth: float = 1.4
    hype_sensitivity_price: float = 1.1
    base_user_growth: float = 0.004
    growth_accel: float = 0.0002
    max_user_growth: float = 0.025
    churn_rate: float = 0.01
    user_carrying_capacity: int = 1_200_000
    adoption_saturation_power: float = 1.6
    adoption_volatility: float = 0.012
    adoption_price_sensitivity: float = 0.45
    churn_price_sensitivity: float = 0.9
    churn_confidence_weight: float = 0.7
    macro_shock_chance: float = 0.04
    macro_shock_magnitude: float = 0.35
    mint_per_new_user: float = 5.0
    mint_price_floor: float = 0.6
    mint_price_ceiling: float = 2.4
    mint_pause_below_price: float = 0.4
    burn_above_price: float = 4.0
    burn_fraction: float = 0.03
    supply_floor: float = 10_000_000.0
    hold_ratio_start: float = 0.18
    hold_ratio_end: float = 0.28
    liquidity_base: float = 2_200_000.0
    liquidity_per_user: float = 45.0
    impact_coeff: float = 0.32
    organic_noise: float = 0.05
    confidence_start: float = 0.55
    confidence_floor: float = 0.25
    confidence_ceiling: float = 0.88
    confidence_sensitivity: float = 2.9
    treasury_seed_tokens: float = 0.0
    treasury_seed_cash: float = 15_000.0
    baseline_reversion: float = 0.12
    arbitrage_flow_strength: float = 0.4
    intrastep_slices: int = 4
    hype_settings: HypeSettings = field(default_factory=HypeSettings)
    supply_plan: TokenSupplyPlan = field(default_factory=TokenSupplyPlan)
    regime_settings: MarketRegimeSettings = field(default_factory=MarketRegimeSettings)
    algorithm_settings: AlgorithmSettings = field(default_factory=AlgorithmSettings)
    random_seed: int = 1337


@dataclass
class PolicySettings:
    enabled: bool = False
    nav_band_soft: float = 0.01
    nav_band_hard: float = 0.03
    fee_rebate_strength: float = 0.4
    fee_penalty_strength: float = 0.6
    omo_strength: float = 0.35
    savings_strength: float = 0.25
    breaker_threshold: float = 0.05
    breaker_flow_shock: float = 0.65
    max_omo_fraction: float = 0.25  # fraction of treasury inventory per step
    gas_subsidy_pool: float = 40_000.0
    gas_subsidy_rate: float = 0.02
    random_attack_mode: str = "off"  # off, light, medium, heavy
    reversion_bonus: float = 0.25
    arb_flow_bonus: float = 0.65
    liquidity_support_strength: float = 0.35
    module_fee_incentives: bool = False
    module_liquidity_support: bool = False
    module_omo: bool = False
    module_policy_arbitrage: bool = False
    module_savings: bool = False
    module_gas_subsidy: bool = False
    module_circuit_breaker: bool = False
    activation_price: float = 0.0
    activation_step: int = 0
    activation_confidence: float = 0.0
    ramp_up_steps: int = 0
    bootstrap_cash: float = 0.0
    bootstrap_tokens: float = 0.0
    financing_contrib_rate: float = 0.0
    financing_pre_stage: bool = False
    bankruptcy_liquidity_hit: float = 0.6
    bankruptcy_confidence_hit: float = 0.3
    bankruptcy_selloff: float = 0.4
    module_mint_control: bool = False
    mint_ramp_strength: float = 0.4
    mint_support_floor: float = 0.6
    mint_support_ceiling: float = 1.2


@dataclass
class SimulationOutput:
    timeline: pd.DataFrame
    attacker_trades: pd.DataFrame
    attacker_state: pd.DataFrame
    config: SimulationConfig
    policy: PolicySettings
    algorithm: AlgorithmSettings
    attacks: List[AttackEvent]
    attacker_settings: "AttackerSettings"


@dataclass
class AttackerSettings:
    auto_enabled: bool = False
    aggression: float = 0.35
    capital: float = 2_000_000.0
    max_step_fraction: float = 0.18
    reinvest_profits: bool = True
    risk_tolerance: float = 0.5
    patience: float = 0.4
    pump_bias: float = 0.5  # 0 dump bias, 1 pump bias
    loss_aversion: float = 0.4
    escalate_on_profit: bool = True
    cooloff_steps: int = 6
    signal_threshold: float = 0.08
    micro_steps: int = 6
    shared_brain_noise: float = 0.25
    objective: Literal["maximize_pnl", "maximize_cash", "maximize_tokens", "destabilize", "custom"] = "maximize_pnl"
    time_awareness: bool = True
    horizon_years: float = 1.5
    target_price: float = 0.0
    trigger_price: float = 0.0
    goal_band: float = 0.18
    hoard_cash_ratio: float = 0.35
    hoard_token_ratio: float = 0.4
    final_push_aggression: float = 1.8
    retreat_drawdown: float = 0.22
    adaptive_sizing: bool = True
    allow_flash_crash: bool = True
    cycle_bias: float = 0.5
    max_inventory_multiple: float = 5.0
    objective_note: str = ""
    mode_label: str = "Adaptive Syndicate"
    manual_directives: List[Dict[str, Any]] = field(default_factory=list)
    engine_version: str = "2.0"
    profiles: List[AttackerProfile] = field(default_factory=_default_attacker_profiles)
    orchestrate_profiles: bool = True


@dataclass
class AttackerRuntimeState:
    label: str
    phase: str = "scout"
    cash: float = 0.0
    tokens: float = 0.0
    cost_basis: float = 0.0
    pnl: float = 0.0
    objective_progress: float = 0.0
    trigger_armed: bool = False
    last_price: float = 0.0
    last_action: str = ""
    cooldown: int = 0


def _build_attacker_states(
    profiles: List[AttackerProfile],
    settings: AttackerSettings,
) -> List[AttackerRuntimeState]:
    states: List[AttackerRuntimeState] = []
    for profile in profiles:
        states.append(
            AttackerRuntimeState(
                label=profile.name,
                cash=profile.capital,
                tokens=0.0,
                cost_basis=0.0,
                pnl=0.0,
                phase="scout",
                last_price=0.0,
            )
        )
    if not states:
        states.append(
            AttackerRuntimeState(
                label=settings.mode_label or "Sovereign Desk",
                cash=settings.capital,
                tokens=0.0,
                cost_basis=0.0,
            )
        )
    return states


def _directive_matches(directive: Dict[str, Any], context: Dict[str, float]) -> bool:
    price = context.get("price")
    step = context.get("step")
    deviation = context.get("deviation")
    momentum = context.get("momentum")
    remaining = context.get("time_remaining")
    trigger = directive.get("trigger", {})
    if not trigger:
        return True
    price_min = trigger.get("price_min")
    price_max = trigger.get("price_max")
    if price_min is not None and price is not None and price < price_min:
        return False
    if price_max is not None and price is not None and price > price_max:
        return False
    step_after = trigger.get("step_after")
    if step_after is not None and step is not None and step < step_after:
        return False
    step_before = trigger.get("step_before")
    if step_before is not None and step is not None and step > step_before:
        return False
    dev_min = trigger.get("deviation_min")
    if dev_min is not None and deviation is not None and deviation < dev_min:
        return False
    dev_max = trigger.get("deviation_max")
    if dev_max is not None and deviation is not None and deviation > dev_max:
        return False
    momentum_min = trigger.get("momentum_min")
    if momentum_min is not None and momentum is not None and momentum < momentum_min:
        return False
    horizon_max = trigger.get("time_remaining_max")
    if horizon_max is not None and remaining is not None and remaining > horizon_max:
        return False
    horizon_min = trigger.get("time_remaining_min")
    if horizon_min is not None and remaining is not None and remaining < horizon_min:
        return False
    return True


def _apply_manual_directives(
    settings: AttackerSettings,
    state: AttackerRuntimeState,
    context: Dict[str, float],
) -> Optional[str]:
    for directive in settings.manual_directives or []:
        scope = directive.get("scope", "any")
        if scope not in ("any", state.label):
            continue
        if not _directive_matches(directive, context):
            continue
        new_phase = directive.get("phase")
        if new_phase:
            return str(new_phase)
    return None


def _attacker_phase_step(
    rng: random.Random,
    settings: AttackerSettings,
    profile: AttackerProfile,
    state: AttackerRuntimeState,
    context: Dict[str, float],
) -> Tuple[float, float, str, Dict[str, Any]]:
    """Return token flow (+ buy, - sell), liquidity penalty, phase, and action metadata."""
    price = context["price"]
    nav_price = context["nav_price"]
    fundamental_price = context["fundamental_price"]
    deviation = context["deviation"]
    momentum = context["momentum"]
    time_remaining = context["time_remaining"]
    total_steps = context["total_steps"]
    step = context["step"]
    liquidity_depth = max(context["liquidity_depth"], 1.0)
    free_float = max(context["free_float"], 1.0)

    liquidity_penalty = 0.0
    meta: Dict[str, Any] = {}

    # Determine target price band
    if settings.target_price > 0:
        target_price = settings.target_price
    else:
        if settings.objective in ("maximize_pnl", "destabilize"):
            target_price = (fundamental_price + nav_price) * 0.5
        elif settings.objective == "maximize_cash":
            target_price = max(nav_price * 1.05, fundamental_price * 1.1)
        else:
            target_price = fundamental_price
    goal_band = max(0.02, settings.goal_band)
    lower_band = target_price * (1 - goal_band)
    upper_band = target_price * (1 + goal_band)

    # Evaluate manual directives first
    directive_phase = _apply_manual_directives(settings, state, context)
    if directive_phase:
        state.phase = directive_phase

    # Automatic phase transitions
    if settings.time_awareness and time_remaining < 0.1 and state.tokens > 0 and settings.objective in ("maximize_cash", "maximize_pnl"):
        state.phase = "exit"
    elif settings.time_awareness and time_remaining < 0.2 and state.cash > state.tokens * price * 0.5 and settings.objective == "maximize_tokens":
        state.phase = "accumulate_tokens"
    elif state.phase == "scout":
        if price < lower_band:
            state.phase = "accumulate_tokens"
        elif price > upper_band and state.tokens > 0:
            state.phase = "dump"
        elif settings.objective == "destabilize":
            state.phase = "probe"
        else:
            state.phase = "range_trade"
    elif settings.trigger_price > 0 and price >= settings.trigger_price and not state.trigger_armed:
        state.phase = "pump"
        state.trigger_armed = True
    elif settings.trigger_price > 0 and price <= settings.trigger_price * (1 - goal_band) and state.trigger_armed:
        state.phase = "dump"

    # Cooldown
    if state.cooldown > 0:
        state.cooldown -= 1
        return 0.0, 0.0, state.phase, {"action": "cooldown"}

    # Compute aggression baseline
    base_aggr = settings.aggression * (0.6 + 0.6 * profile.aggression)
    base_aggr *= 0.8 + 0.4 * profile.skill
    if settings.adaptive_sizing:
        pnl_bias = np.tanh(state.pnl / max(profile.capital, 1.0))
        base_aggr *= 1.0 + pnl_bias * settings.risk_tolerance * 0.6
    if settings.objective == "destabilize":
        base_aggr *= 1.2
    base_aggr = float(np.clip(base_aggr, 0.02, 3.0))

    # Determine directional intent
    intent = 0.0
    phase = state.phase
    action_label = "idle"
    pump_push = settings.final_push_aggression if settings.time_awareness and time_remaining < 0.15 else 1.0

    if phase == "accumulate_tokens":
        value_signal = np.tanh((target_price - price) / max(target_price, 1e-6))
        intent = base_aggr * (0.6 + 0.4 * profile.skill) * value_signal
        action_label = "accumulate"
    elif phase == "accumulate_cash":
        intent = -base_aggr * (0.5 + profile.patience * 0.4)
        action_label = "raise_cash"
    elif phase == "pump":
        intent = base_aggr * pump_push * (0.9 + profile.skill * 0.4)
        intent += settings.cycle_bias * 0.2
        action_label = "pump"
    elif phase == "dump":
        intent = -base_aggr * pump_push * (0.8 + profile.skill * 0.4)
        intent -= (1 - settings.cycle_bias) * 0.2
        action_label = "dump"
    elif phase == "exit":
        intent = -base_aggr * (1.2 + profile.skill * 0.3)
        action_label = "exit"
    elif phase == "probe":
        intent = base_aggr * np.sign(rng.gauss(0, 1)) * 0.6
        action_label = "probe"
    elif phase == "range_trade":
        mean_revert = np.tanh(-deviation * 2.5)
        intent = base_aggr * (0.4 + profile.skill * 0.4) * mean_revert
        intent += momentum * 0.2
        action_label = "range"
    else:
        intent = base_aggr * (momentum * 0.6 - deviation * 0.4)
        action_label = "opportunistic"

    # Add randomness
    noise = rng.gauss(0.0, settings.shared_brain_noise * 0.15)
    intent += noise

    # Convert intent to tokens, respecting inventory and capital limits
    max_inventory_value = profile.capital * max(profile.leverage, 1.0) * max(settings.max_inventory_multiple, 1.0)
    position_value = state.tokens * price
    portfolio_value = state.cash + position_value
    max_position_value = min(max_inventory_value, portfolio_value + state.cash)

    step_budget_value = settings.max_step_fraction * max(profile.capital, portfolio_value + 1e-6)
    step_budget_value = max(step_budget_value, settings.capital * 0.01 if settings.capital else 0.0)

    if intent > 0:
        max_cash_to_spend = min(state.cash, step_budget_value)
        desired_allocation = settings.hoard_token_ratio if settings.objective in ("maximize_tokens",) else settings.hoard_token_ratio * 0.5
        target_tokens_value = max_position_value * desired_allocation
        if position_value >= target_tokens_value and settings.objective != "destabilize":
            intent *= 0.4
        buy_tokens = max_cash_to_spend / max(price, 1e-6) * float(np.clip(intent, 0.0, 3.0))
        buy_tokens = float(np.clip(buy_tokens, 0.0, max_cash_to_spend / max(price, 1e-6)))
        state.cash -= buy_tokens * price
        state.tokens += buy_tokens
        if state.tokens > 0:
            state.cost_basis = (state.cost_basis * (state.tokens - buy_tokens) + buy_tokens * price) / max(state.tokens, 1e-6)
        liquidity_penalty = min(0.35, buy_tokens * price / max(liquidity_depth, 1.0) * 0.5)
        state.last_action = "buy"
        token_flow = buy_tokens
    elif intent < 0:
        max_tokens_to_sell = min(state.tokens, step_budget_value / max(price, 1e-6) * (1.2 if phase in ("dump", "exit") else 1.0))
        if settings.objective == "maximize_cash" and time_remaining < 0.25:
            max_tokens_to_sell = state.tokens
        sell_tokens = max_tokens_to_sell * float(np.clip(abs(intent), 0.0, 3.0))
        sell_tokens = float(np.clip(sell_tokens, 0.0, state.tokens))
        state.tokens -= sell_tokens
        state.cash += sell_tokens * price
        realized = (price - state.cost_basis) * sell_tokens
        state.pnl += realized
        liquidity_penalty = min(0.45, sell_tokens * price / max(liquidity_depth, 1.0) * 0.6)
        state.last_action = "sell"
        token_flow = -sell_tokens
        if state.tokens <= 1e-6:
            state.cost_basis = price
    else:
        token_flow = 0.0
        state.last_action = "idle"

    # Update phase heuristics
    if phase in ("pump", "dump") and step_budget_value > 0:
        state.cooldown = max(state.cooldown, int(max(1, settings.cooloff_steps * (0.6 + 0.4 * (1 - profile.patience)))))
    if phase == "exit" and state.tokens <= 1e-6:
        state.phase = "park"
    elif phase == "accumulate_tokens" and price >= target_price:
        state.phase = "range_trade"
    elif phase == "accumulate_cash" and state.cash >= profile.capital * settings.hoard_cash_ratio:
        state.phase = "pump"

    state.last_price = price
    meta.update(
        {
            "phase": state.phase,
            "intent": intent,
            "action": action_label,
            "cash": state.cash,
            "tokens": state.tokens,
            "pnl": state.pnl,
        }
    )
    return token_flow, liquidity_penalty, state.phase, meta


def _algorithm_metric_value(metric: str, metrics: Dict[str, float]) -> float:
    if metric == "price":
        return metrics.get("price", 0.0)
    if metric == "peg_deviation":
        return metrics.get("deviation", 0.0)
    if metric == "growth":
        return metrics.get("growth", 0.0)
    if metric == "volatility":
        return metrics.get("volatility", 0.0)
    if metric == "treasury_nav":
        return metrics.get("treasury_nav_gap", 0.0)
    if metric == "custom_index":
        return metrics.get("custom_index", 0.0)
    return metrics.get(metric, 0.0)


def _compute_objective_signal(
    objective: AlgorithmObjective,
    metrics: Dict[str, float],
) -> Tuple[float, float]:
    """Return (signal, satisfaction) for an objective."""
    value = _algorithm_metric_value(objective.metric, metrics)
    target = objective.target_value
    tolerance = max(objective.tolerance, 1e-6)
    diff = value - target
    normalized = diff / tolerance

    if objective.comparison == "track":
        signal = -np.tanh(normalized)
    elif objective.comparison == "at_most":
        normalized = max(0.0, normalized)
        signal = -np.tanh(normalized)
    elif objective.comparison == "at_least":
        normalized = min(0.0, normalized)
        signal = -np.tanh(normalized)
    elif objective.comparison == "maximize":
        scale = max(abs(target) + tolerance, tolerance)
        signal = np.tanh(value / scale)
    elif objective.comparison == "minimize":
        scale = max(abs(target) + tolerance, tolerance)
        signal = -np.tanh(value / scale)
    else:
        signal = -np.tanh(normalized)

    if objective.dynamic_trigger is not None:
        trigger = objective.dynamic_trigger
        if (diff > 0 and trigger >= 0 and value >= trigger) or (diff < 0 and trigger <= 0 and value <= trigger):
            signal *= 1.5

    horizon_scale = 1.0 / max(objective.horizon_years, 0.1)
    signal *= objective.weight * (0.6 + 0.4 * horizon_scale)

    satisfaction = 1.0 - min(1.0, abs(diff) / (abs(target) + tolerance * 5))
    satisfaction = max(0.0, float(satisfaction))
    return float(signal), satisfaction


def render_sidebar_guide() -> None:
    with st.sidebar:
        st.header("Quick Start")
        st.markdown(
            "1. **Base Market Conditions** – configure users, liquidity, supply, and hype.\n"
            "2. **Algorithm Forge** – choose objectives, enable modules, and set treasury autopilot.\n"
            "3. **Attacker Lab** – add scripted or autonomous adversaries once the baseline looks healthy.\n"
            "4. **Simulation** – run the scenario, read the charts, and loop back with tweaks."
        )
        st.markdown(
            "Goal: keep `Peg Deviation %` near zero while maintaining healthy treasury inventories and user growth."
        )
        with st.expander("Reference: key concepts", expanded=False):
            st.markdown(
                "**Peg** – target price derived from gold backing.  \n"
                "**NAV** – net asset value per token; your anchor.  \n"
                "**Peg deviation** – market price minus NAV, shown as a percentage.  \n"
                "**Treasury NAV** – cash + tokens owned by the protocol.  \n"
                "**Liquidity depth** – capital in the order book; higher depth resists price shocks.  \n"
                "**Autopilot** – fee skim + buyback logic that manages treasury cash automatically.  \n"
                "**Attack modes** – scripted or AI desks that stress-test your setup."
            )
        with st.expander("Reference: workflow tips", expanded=False):
            st.markdown(
                "- Run once with all automation off to understand the base market.  \n"
                "- Add one intervention at a time and re-run so you can see cause and effect.  \n"
                "- Use light background attacks before jumping to heavy scenarios.  \n"
                "- Export results from the Simulation tab if you need to compare runs offline."
            )


def render_help_expander() -> None:
    with st.expander("❓ Help & Getting Started", expanded=False):
        st.markdown("### What is this simulator?")
        st.markdown(
            "This is a playground for testing whether a token pegged to gold can survive in the real world. "
            "You control everything: how users behave, how the market works, what defense mechanisms you use, "
            "and how attackers try to break your peg. Your goal is to keep the market price close to the gold-backed value."
        )
        
        st.markdown("### Your Goal")
        st.markdown(
            "**Keep the peg stable!** You want your token's market price to track the gold price (NAV) as closely as possible. "
            "Watch the **Peg Deviation** chart in the Simulation tab—that's your report card. The closer to 0%, the better you're doing."
        )
        
        st.markdown("### Step-by-Step Workflow")
        st.markdown(
            "**1️⃣ Base Market Conditions Tab**\n"
            "- Set up the world: How many users do you start with? How fast do they join?\n"
            "- Configure supply: How many tokens exist? When do they unlock?\n"
            "- Define behavior: How sticky are users? How much do they hold vs. trade?\n"
            "- **Tip:** Start with defaults, run once to see what happens, then tweak.\n\n"
            
            "**2️⃣ Algorithm Forge Tab**\n"
            "- Design your defense: Should the system buy when price drops? Sell when it rises?\n"
            "- Set objectives: Track the peg? Minimize volatility? Grow users?\n"
            "- Enable modules: Circuit breakers, fee incentives, automatic mint/burn\n"
            "- **Tip:** Leave everything OFF for your first run to see the 'natural' market. Then turn on defenses.\n\n"
            
            "**3️⃣ Attacker Lab Tab**\n"
            "- Add pressure: Script attacks or enable AI attackers that try to profit by breaking your peg\n"
            "- Configure goals: Should they pump? Dump? Just try to make money?\n"
            "- Test resilience: Can your defenses survive coordinated attacks?\n"
            "- **Tip:** Start with attacks OFF, get your baseline working, then add light attacks.\n\n"
            
            "**4️⃣ Simulation Tab**\n"
            "- Hit the big **Run Simulation** button\n"
            "- Watch your token over months/years\n"
            "- Check if the peg held, if you went bankrupt, if attackers made money\n"
            "- **Iterate:** Go back to other tabs, adjust, re-run until it works!"
        )
        
        st.markdown("### Reading the Charts")
        st.markdown(
            "**Prices vs Anchors** - The lines should stay close together. If they spread apart, your peg is breaking.\n\n"
            "**Peg Deviation %** - This is THE metric. ±2% is pretty good. ±10%+ means you have problems.\n\n"
            "**Treasury Inventories** - Don't let these hit zero! If you run out of tokens or cash, you can't defend anymore.\n\n"
            "**Users & Liquidity** - More users = more demand. More liquidity = more stability.\n\n"
            "**Attacker Summary** - Look at their final PnL (profit/loss). If it's hugely positive, they won.\n"
        )
        
        st.markdown("### Common Beginner Mistakes")
        st.markdown(
            "❌ **Running out of treasury funds** - If your algorithm is too aggressive, it'll spend all its ammo early.\n\n"
            "❌ **Setting impossible goals** - Don't expect a perfect 0% deviation 24/7. Real markets wiggle.\n\n"
            "❌ **Ignoring user behavior** - If everyone sells when price dips, no algorithm can save you.\n\n"
            "❌ **Testing with attacks first** - Get your baseline working before adding attackers.\n\n"
            "❌ **Too many modules at once** - Start simple, add complexity slowly.\n"
        )
        
        st.markdown("### Quick Experiments to Try")
        st.markdown(
            "🧪 **Experiment 1: Baseline Reality**\n"
            "- Keep all defaults in Base Market Conditions\n"
            "- Turn OFF algorithm and attackers\n"
            "- Run 5 years and see what happens naturally\n\n"
            
            "🧪 **Experiment 2: Simple Defense**\n"
            "- Use your baseline from Experiment 1\n"
            "- In Algorithm Forge, enable the algorithm with 'defend' mode\n"
            "- Turn on just 'Circuit Breaker' and 'OMO' modules\n"
            "- See if it helps stabilize\n\n"
            
            "🧪 **Experiment 3: Under Attack**\n"
            "- Use your defended setup from Experiment 2\n"
            "- In Attacker Lab, enable AI desks with 'maximize_pnl' objective\n"
            "- Give them USD 1M capital at 0.5 aggression\n"
            "- Can your defenses survive?\n"
        )
        
        st.markdown("### Still Confused?")
        st.markdown(
            "**Start here:**\n"
            "1. Click the **📖 Beginner's Glossary** in the sidebar to learn the terms\n"
            "2. Go to Simulation tab and just hit 'Run simulation' with all defaults\n"
            "3. Look at the Peg Deviation chart—is it stable or wild?\n"
            "4. Go to Base Market Conditions and change ONE thing (like user growth)\n"
            "5. Run again and see what changed\n\n"
            
            "You'll learn by doing! Don't worry about breaking things—that's the whole point of a simulator."
        )
        st.caption("💬 Remember: This is a learning tool. Real tokenomics are even messier. But if you can't make it work here, it won't work in reality.")
        st.markdown("---")
        st.markdown(
            "**Think of it like a video game:**\n"
            "- Base Market = Character stats and world settings\n"
            "- Algorithm = Your special abilities and power-ups\n"
            "- Attackers = Enemy bosses trying to beat you\n"
            "- Simulation = Actually playing the game\n\n"
            "Your high score is how stable your peg stays! 🎮"
        )


def _plotly(fig, full_width: bool = True, key: Optional[str] = None) -> None:
    """Render Plotly figures with the appropriate Streamlit API, future-proofing width handling."""
    kwargs: Dict[str, Any] = {"config": PLOTLY_CONFIG, "theme": "streamlit"}
    if PLOTLY_SUPPORTS_WIDTH:
        kwargs["width"] = "stretch" if full_width else "content"
    else:
        kwargs["use_container_width"] = full_width
    effective_key = key or f"plotly-chart-{next(_plotly_key_counter)}"
    st.plotly_chart(fig, key=effective_key, **kwargs)


def _annual_to_step_rate(annual_rate: float, steps_per_year: int) -> float:
    return (1 + annual_rate) ** (1 / steps_per_year) - 1


def _step_to_annual_rate(step_rate: float, steps_per_year: int) -> float:
    if steps_per_year <= 0:
        return step_rate
    return (1 + step_rate) ** steps_per_year - 1


def _setup_random(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    return rng


def _generate_random_attacks(
    rng: random.Random,
    total_steps: int,
    mode: str,
) -> List[AttackEvent]:
    if mode == "off":
        return []
    if mode == "light":
        count = rng.randint(2, 3)
    elif mode == "heavy":
        count = rng.randint(6, 9)
    else:
        count = rng.randint(4, 6)

    attacks: List[AttackEvent] = []
    for i in range(count):
        start = rng.randint(4, max(5, total_steps - 6))
        duration = rng.randint(2, 6)
        side = rng.choice(["buy", "sell", "mixed"])
        magnitude = rng.uniform(0.04, 0.16)
        liquidity_drop = rng.uniform(0.2, 0.6) if side == "mixed" else rng.uniform(0.0, 0.35)
        attacks.append(
            AttackEvent(
                name=f"Random-{i + 1}",
                start_step=start,
                duration=duration,
                side=side,
                magnitude=magnitude,
                liquidity_drop=liquidity_drop,
                notes="auto-generated",
            )
        )
    return attacks


def _coerce_regime_settings(regime_settings: Any) -> MarketRegimeSettings:
    defaults = MarketRegimeSettings()
    if isinstance(regime_settings, MarketRegimeSettings):
        for f in fields(MarketRegimeSettings):
            if not hasattr(regime_settings, f.name):
                setattr(regime_settings, f.name, getattr(defaults, f.name))
        return regime_settings
    if isinstance(regime_settings, dict):
        payload = {f.name: regime_settings.get(f.name, getattr(defaults, f.name)) for f in fields(MarketRegimeSettings)}
        return MarketRegimeSettings(**payload)
    return defaults


def _coerce_algorithm_settings(algorithm_settings: Any) -> AlgorithmSettings:
    defaults = AlgorithmSettings()
    if isinstance(algorithm_settings, AlgorithmSettings):
        # Patch missing attributes
        for f in fields(AlgorithmSettings):
            if not hasattr(algorithm_settings, f.name):
                setattr(algorithm_settings, f.name, getattr(defaults, f.name))
        if getattr(algorithm_settings, "custom_params", None) is None:
            algorithm_settings.custom_params = {}
        # Objectives/modules may be dicts
        objectives: List[AlgorithmObjective] = []
        for obj in getattr(algorithm_settings, "objectives", []) or []:
            if isinstance(obj, AlgorithmObjective):
                objectives.append(obj)
            elif isinstance(obj, dict):
                defaults_obj = AlgorithmObjective()
                payload = {f.name: obj.get(f.name, getattr(defaults_obj, f.name)) for f in fields(AlgorithmObjective)}
                objectives.append(AlgorithmObjective(**payload))
        if not objectives:
            objectives = [AlgorithmObjective()]
        algorithm_settings.objectives = objectives
        modules = getattr(algorithm_settings, "modules", None)
        if isinstance(modules, AlgorithmModuleSettings):
            for f in fields(AlgorithmModuleSettings):
                if not hasattr(modules, f.name):
                    setattr(modules, f.name, getattr(AlgorithmModuleSettings(), f.name))
        elif isinstance(modules, dict):
            defaults_modules = AlgorithmModuleSettings()
            payload = {f.name: modules.get(f.name, getattr(defaults_modules, f.name)) for f in fields(AlgorithmModuleSettings)}
            algorithm_settings.modules = AlgorithmModuleSettings(**payload)
        else:
            algorithm_settings.modules = AlgorithmModuleSettings()
        crash_defense = getattr(algorithm_settings, "crash_defense", None)
        if isinstance(crash_defense, CrashDefenseSettings):
            defaults_crash = CrashDefenseSettings()
            for f in fields(CrashDefenseSettings):
                if not hasattr(crash_defense, f.name):
                    setattr(crash_defense, f.name, getattr(defaults_crash, f.name))
        elif isinstance(crash_defense, dict):
            defaults_crash = CrashDefenseSettings()
            payload = {f.name: crash_defense.get(f.name, getattr(defaults_crash, f.name)) for f in fields(CrashDefenseSettings)}
            algorithm_settings.crash_defense = CrashDefenseSettings(**payload)
        else:
            algorithm_settings.crash_defense = CrashDefenseSettings()
        return algorithm_settings
    if isinstance(algorithm_settings, dict):
        defaults_modules = AlgorithmModuleSettings()
        modules_dict = algorithm_settings.get("modules", {})
        if isinstance(modules_dict, dict):
            modules_payload = {f.name: modules_dict.get(f.name, getattr(defaults_modules, f.name)) for f in fields(AlgorithmModuleSettings)}
            modules = AlgorithmModuleSettings(**modules_payload)
        else:
            modules = AlgorithmModuleSettings()
        defaults_crash = CrashDefenseSettings()
        crash_dict = algorithm_settings.get("crash_defense", {})
        if isinstance(crash_dict, dict):
            crash_payload = {f.name: crash_dict.get(f.name, getattr(defaults_crash, f.name)) for f in fields(CrashDefenseSettings)}
            crash_defense = CrashDefenseSettings(**crash_payload)
        else:
            crash_defense = CrashDefenseSettings()
        objectives_list = []
        for obj in algorithm_settings.get("objectives", []) or []:
            if isinstance(obj, dict):
                defaults_obj = AlgorithmObjective()
                payload = {f.name: obj.get(f.name, getattr(defaults_obj, f.name)) for f in fields(AlgorithmObjective)}
                objectives_list.append(AlgorithmObjective(**payload))
        if not objectives_list:
            objectives_list = [AlgorithmObjective()]
        payload = {f.name: algorithm_settings.get(f.name, getattr(defaults, f.name)) for f in fields(AlgorithmSettings) if f.name not in {"modules", "objectives", "crash_defense"}}
        final = AlgorithmSettings(**payload)
        final.modules = modules
        final.objectives = objectives_list
        final.crash_defense = crash_defense
        return final
    return defaults


def _apply_playbook_presets(algo_cfg: AlgorithmSettings, mode: str) -> None:
    mode = mode or "defend"
    if not hasattr(algo_cfg, "modules"):
        algo_cfg.modules = AlgorithmModuleSettings()
    if not hasattr(algo_cfg, "crash_defense"):
        algo_cfg.crash_defense = CrashDefenseSettings()
    if not hasattr(algo_cfg, "custom_params") or algo_cfg.custom_params is None:
        algo_cfg.custom_params = {}

    modules = algo_cfg.modules
    crash_cfg = algo_cfg.crash_defense
    autopilot_cfg = dict(algo_cfg.custom_params.get("growth_autopilot", {}))

    base_autopilot = {
        "enabled": False,
        "fee_capture_rate": autopilot_cfg.get("fee_capture_rate", 0.12),
        "override_financing": autopilot_cfg.get("override_financing", False),
        "cash_tolerance_pct": autopilot_cfg.get("cash_tolerance_pct", 1.5),
        "min_tax_pct": autopilot_cfg.get("min_tax_pct", 1.2),
        "max_tax_pct": autopilot_cfg.get("max_tax_pct", 6.0),
        "fee_step_pct": autopilot_cfg.get("fee_step_pct", 0.35),
        "buy_push": autopilot_cfg.get("buy_push", 0.6),
        "buy_brake": autopilot_cfg.get("buy_brake", 0.5),
        "cash_buffer_pct": autopilot_cfg.get("cash_buffer_pct", 40.0),
        "surplus_spend_rate_pct": autopilot_cfg.get("surplus_spend_rate_pct", 20.0),
        "oracle_enabled": autopilot_cfg.get("oracle_enabled", False),
        "oracle_horizon_steps": autopilot_cfg.get("oracle_horizon_steps", 6),
        "oracle_goal": autopilot_cfg.get("oracle_goal", "market_cap"),
        "target_metric": autopilot_cfg.get("target_metric", algo_cfg.custom_params.get("primary_goal_mode", "market_cap")),
        "oracle_weight": autopilot_cfg.get("oracle_weight", 1.0),
        "oracle_accuracy": autopilot_cfg.get("oracle_accuracy", 0.65),
    }

    if mode == "defend":
        algo_cfg.objectives = [
            AlgorithmObjective(
                name="Hold the peg",
                metric="peg_deviation",
                target_value=0.0,
                tolerance=0.015,
                comparison="track",
                weight=1.6,
                horizon_years=0.2,
            ),
            AlgorithmObjective(
                name="Kill volatility",
                metric="volatility",
                target_value=0.02,
                tolerance=0.05,
                comparison="at_most",
                weight=1.1,
                horizon_years=0.4,
            ),
            AlgorithmObjective(
                name="Protect treasury",
                metric="treasury_nav",
                target_value=0.0,
                tolerance=0.1,
                comparison="maximize",
                weight=0.9,
                horizon_years=0.6,
            ),
        ]
        modules.circuit_breaker = True
        modules.circuit_threshold = 0.06
        modules.circuit_cooldown_steps = 6
        modules.adaptive_mint = False
        modules.adaptive_burn = True
        modules.liquidity_support = True
        modules.alpha_stabilizer = True
        modules.trend_follow = False
        modules.drawdown_guard = True
        modules.drawdown_threshold = 0.16
        modules.velocity_targets = True
        modules.objective_lock_when_met = True
        crash_cfg.enabled = True
        crash_cfg.drop_threshold_pct = 10.0
        crash_cfg.detection_window = 4
        crash_cfg.resource_commitment_pct = 55.0
        crash_cfg.aggression_pct = 70.0
        crash_cfg.gas_subsidy_share_pct = 35.0
        crash_cfg.circuit_lock_steps = 8
        crash_cfg.stabilize_steps = 6
        crash_cfg.cooldown_steps = 14
        crash_cfg.gas_efficiency = 0.45
        base_autopilot.update(
            {
                "enabled": False,
                "fee_capture_rate": 0.14,
                "cash_tolerance_pct": 1.6,
                "min_tax_pct": 0.9,
                "max_tax_pct": 4.8,
                "buy_push": 0.45,
                "buy_brake": 0.8,
                "cash_buffer_pct": 45.0,
                "surplus_spend_rate_pct": 18.0,
                "oracle_enabled": False,
                "target_metric": "market_cap",
            }
        )
        algo_cfg.custom_params["primary_goal_mode"] = "market_cap"
    elif mode == "grow":
        algo_cfg.objectives = [
            AlgorithmObjective(
                name="Pump price",
                metric="price",
                target_value=1.0,
                tolerance=0.3,
                comparison="maximize",
                weight=1.4,
                horizon_years=0.5,
            ),
            AlgorithmObjective(
                name="Expand users",
                metric="growth",
                target_value=0.012,
                tolerance=0.015,
                comparison="maximize",
                weight=1.0,
                horizon_years=0.4,
            ),
            AlgorithmObjective(
                name="Tame drawdowns",
                metric="peg_deviation",
                target_value=0.0,
                tolerance=0.04,
                comparison="track",
                weight=0.8,
                horizon_years=0.25,
            ),
        ]
        modules.circuit_breaker = True
        modules.circuit_threshold = 0.09
        modules.circuit_cooldown_steps = 4
        modules.adaptive_mint = True
        modules.adaptive_burn = True
        modules.liquidity_support = True
        modules.alpha_stabilizer = False
        modules.trend_follow = True
        modules.drawdown_guard = True
        modules.drawdown_threshold = 0.2
        modules.velocity_targets = False
        modules.objective_lock_when_met = False
        crash_cfg.enabled = True
        crash_cfg.drop_threshold_pct = 15.0
        crash_cfg.detection_window = 5
        crash_cfg.resource_commitment_pct = 45.0
        crash_cfg.aggression_pct = 60.0
        crash_cfg.gas_subsidy_share_pct = 22.0
        crash_cfg.circuit_lock_steps = 6
        crash_cfg.stabilize_steps = 5
        crash_cfg.cooldown_steps = 10
        crash_cfg.gas_efficiency = 0.4
        base_autopilot.update(
            {
                "enabled": True,
                "fee_capture_rate": 0.18,
                "cash_tolerance_pct": 1.2,
                "min_tax_pct": 0.7,
                "max_tax_pct": 5.5,
                "buy_push": 0.95,
                "buy_brake": 0.4,
                "cash_buffer_pct": 28.0,
                "surplus_spend_rate_pct": 34.0,
                "oracle_enabled": True,
                "oracle_goal": "token_price",
                "target_metric": "token_price",
                "oracle_weight": 1.8,
                "oracle_accuracy": 0.75,
                "oracle_horizon_steps": 8,
            }
        )
        algo_cfg.custom_params["primary_goal_mode"] = "token_price"
    elif mode == "stabilize":
        algo_cfg.objectives = [
            AlgorithmObjective(
                name="Smooth volatility",
                metric="volatility",
                target_value=0.015,
                tolerance=0.035,
                comparison="at_most",
                weight=1.2,
                horizon_years=0.35,
            ),
            AlgorithmObjective(
                name="Neutral drift",
                metric="velocity",
                target_value=0.0,
                tolerance=0.02,
                comparison="track",
                weight=0.9,
                horizon_years=0.4,
            ),
            AlgorithmObjective(
                name="Backstop peg",
                metric="peg_deviation",
                target_value=0.0,
                tolerance=0.025,
                comparison="track",
                weight=1.0,
                horizon_years=0.3,
            ),
        ]
        modules.circuit_breaker = True
        modules.circuit_threshold = 0.05
        modules.circuit_cooldown_steps = 5
        modules.adaptive_mint = False
        modules.adaptive_burn = True
        modules.liquidity_support = True
        modules.alpha_stabilizer = True
        modules.trend_follow = False
        modules.drawdown_guard = True
        modules.drawdown_threshold = 0.14
        modules.velocity_targets = True
        modules.objective_lock_when_met = True
        crash_cfg.enabled = True
        crash_cfg.drop_threshold_pct = 8.0
        crash_cfg.detection_window = 6
        crash_cfg.resource_commitment_pct = 40.0
        crash_cfg.aggression_pct = 55.0
        crash_cfg.gas_subsidy_share_pct = 40.0
        crash_cfg.circuit_lock_steps = 7
        crash_cfg.stabilize_steps = 7
        crash_cfg.cooldown_steps = 16
        crash_cfg.gas_efficiency = 0.5
        base_autopilot.update(
            {
                "enabled": True,
                "fee_capture_rate": 0.12,
                "cash_tolerance_pct": 1.5,
                "min_tax_pct": 0.8,
                "max_tax_pct": 4.5,
                "buy_push": 0.55,
                "buy_brake": 0.6,
                "cash_buffer_pct": 38.0,
                "surplus_spend_rate_pct": 22.0,
                "oracle_enabled": True,
                "oracle_goal": "hybrid",
                "target_metric": "hybrid",
                "oracle_weight": 1.2,
                "oracle_accuracy": 0.68,
                "oracle_horizon_steps": 7,
            }
        )
        algo_cfg.custom_params["primary_goal_mode"] = "hybrid"
    else:
        return

    algo_cfg.custom_params.setdefault("growth_autopilot", {})
    algo_cfg.custom_params["growth_autopilot"].update(base_autopilot)
    algo_cfg.modules = modules
    algo_cfg.crash_defense = crash_cfg
def _prepare_attacks(manual_events: List[Dict[str, Any]]) -> List[AttackEvent]:
    attacks: List[AttackEvent] = []
    for row in manual_events:
        try:
            attacks.append(
                AttackEvent(
                    name=str(row.get("name", "Attack")),
                    start_step=int(row.get("start_step", 0)),
                    duration=max(1, int(row.get("duration", 1))),
                    side=str(row.get("side", "sell")),
                    magnitude=float(row.get("magnitude", 0.05)),
                    notes=str(row.get("notes", "")),
                    liquidity_drop=float(row.get("liquidity_drop", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return attacks


def run_simulation(
    sim: SimulationConfig,
    policy: PolicySettings,
    manual_attacks: List[Dict[str, Any]],
    attacks_enabled: bool,
    attacker_settings: AttackerSettings,
) -> SimulationOutput:
    rng = _setup_random(sim.random_seed)
    total_steps = sim.years * sim.steps_per_year
    step_rate_gold = _annual_to_step_rate(sim.gold_drift_annual, sim.steps_per_year)
    step_vol_gold = sim.gold_vol_annual / math.sqrt(sim.steps_per_year)

    if attacks_enabled:
        manual_attack_events = _prepare_attacks(manual_attacks)
        random_attack_events = _generate_random_attacks(rng, total_steps, policy.random_attack_mode)
        attack_events = manual_attack_events + random_attack_events
    else:
        manual_attack_events = []
        random_attack_events = []
        attack_events = []

    epsilon = 1e-5
    df_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    attacker_state_rows: List[Dict[str, Any]] = []

    supply_plan = sim.supply_plan
    hype_cfg = sim.hype_settings
    regime_cfg = _coerce_regime_settings(getattr(sim, "regime_settings", None))
    algorithm_cfg = _coerce_algorithm_settings(getattr(sim, "algorithm_settings", None))

    capacity_baseline = max(float(sim.user_carrying_capacity), float(sim.initial_users))
    capacity_shift = 0.0
    innovation_level = 0.0
    regime_state: str = "neutral"
    regime_timer = 0
    regime_intensity = 0.0

    def sample_regime_state() -> Tuple[str, int, float]:
        if not regime_cfg.enabled:
            neutral_steps = max(1, sim.steps_per_year)
            return "neutral", neutral_steps, 0.0
        roll = rng.random()
        expansion_cut = float(np.clip(regime_cfg.expansion_bias, 0.0, 1.0))
        neutral_pool = float(np.clip(regime_cfg.neutral_chance, 0.0, 1.0 - expansion_cut))
        if roll < expansion_cut:
            state = "expansion"
        elif roll < expansion_cut + neutral_pool:
            state = "neutral"
        else:
            state = "contraction"
        duration_years = max(0.5, rng.gauss(regime_cfg.mean_duration_years, regime_cfg.duration_jitter_years))
        duration_steps = max(1, int(duration_years * max(sim.steps_per_year, 1)))
        intensity = rng.uniform(0.5, 1.0)
        return state, duration_steps, intensity

    if regime_cfg.enabled:
        regime_state, regime_timer, regime_intensity = sample_regime_state()

    gold_price = sim.initial_gold_price
    market_price = max(sim.initial_token_price, epsilon)
    nav_price = gold_price

    fundamental_price = max(sim.intrinsic_value_start, epsilon)
    hype_index = sim.hype_start
    demand_memory = max(market_price, epsilon)

    base_circulating_options = [supply_plan.initial_circulating, sim.initial_free_float, sim.supply_floor]
    free_float = max(option for option in base_circulating_options if option is not None)
    locked_supply = max(supply_plan.initial_locked, 0.0)
    legacy_treasury = sim.treasury_seed_tokens if sim.treasury_seed_tokens > 0 else sim.initial_treasury_tokens
    treasury_tokens = max(supply_plan.initial_treasury, legacy_treasury, 0.0)
    treasury_cash = sim.treasury_seed_cash
    launch_release = 0.0
    if supply_plan.launch_float_release > 0 and locked_supply > 0:
        launch_release = min(locked_supply, locked_supply * supply_plan.launch_float_release)
        locked_supply -= launch_release
        free_float += launch_release
    total_supply = free_float + locked_supply + treasury_tokens
    burn_vault_tokens = 0.0
    burn_vault_released = 0.0
    emission_rate = max(supply_plan.baseline_emission_per_user, 0.0)
    target_emission_rate = max(supply_plan.goal_emission_per_user, emission_rate)
    halving_clock = 0
    next_halving_step = max(supply_plan.halving_interval_steps, 1)
    inflation_guard_timer = 0
    last_goal_hit_step: Optional[int] = None

    users = max(sim.initial_users, 25)
    base_hold_share = np.clip(sim.hold_ratio_start, 0.01, 0.6)
    user_tokens = free_float * base_hold_share
    confidence = np.clip(sim.confidence_start, sim.confidence_floor, sim.confidence_ceiling)
    sticky_users = users * 0.35

    gas_subsidy_pool = policy.gas_subsidy_pool
    savings_pool = 0.0

    minted_cumulative = 0.0
    burned_cumulative = 0.0

    profile_bundle: List[AttackerProfile] = list(attacker_settings.profiles or [])
    if attacker_settings.auto_enabled:
        base_style: Literal["momentum", "pump_and_dump", "liquidity_sapper", "arb_sniper", "scripted"]
        base_style = "pump_and_dump" if attacker_settings.pump_bias >= 0.55 else "momentum"
        base_profile = AttackerProfile(
            name="Primary Syndicate",
            style=base_style,
            capital=attacker_settings.capital,
            aggression=attacker_settings.aggression,
            skill=np.clip(attacker_settings.risk_tolerance, 0.15, 0.95),
            patience=np.clip(attacker_settings.patience, 0.05, 0.95),
            perception=np.clip(1.0 - attacker_settings.signal_threshold, 0.05, 0.95),
            reinvest=attacker_settings.reinvest_profits,
            leverage=1.0 + attacker_settings.max_step_fraction * 1.5,
        )
        profile_bundle = [base_profile] + profile_bundle
    if not profile_bundle:
        profile_bundle = _default_attacker_profiles()

    attacker_states = _build_attacker_states(profile_bundle, attacker_settings)
    attacker_base_capital = sum(profile.capital for profile in profile_bundle)
    algorithm_state = {"circuit_cooldown": 0, "goal_progress": 0.0, "last_signal": 0.0, "algo_flow_memory": 0.0}
    algo_modules = algorithm_cfg.modules if hasattr(algorithm_cfg, "modules") else AlgorithmModuleSettings()

    growth_auto_cfg = algorithm_cfg.custom_params.get("growth_autopilot", {}) if hasattr(algorithm_cfg, "custom_params") else {}
    growth_auto_enabled = bool(growth_auto_cfg.get("enabled", False))
    growth_fee_capture_rate = float(np.clip(growth_auto_cfg.get("fee_capture_rate", 0.12), 0.0, 0.5))
    growth_override_financing = bool(growth_auto_cfg.get("override_financing", False))
    growth_cash_tolerance = max(0.0, float(growth_auto_cfg.get("cash_tolerance_pct", 1.5)) / 100.0)
    growth_min_tax = float(growth_auto_cfg.get("min_tax_pct", max(supply_plan.tx_tax_rate, 0.0) * 100.0)) / 100.0
    growth_max_tax = float(growth_auto_cfg.get("max_tax_pct", max(supply_plan.tx_tax_rate * 100.0 + 3.0, 8.0))) / 100.0
    growth_min_tax = float(np.clip(growth_min_tax, 0.0, 0.25))
    growth_max_tax = float(np.clip(growth_max_tax, 0.0, 0.25))
    if growth_max_tax < growth_min_tax:
        growth_max_tax, growth_min_tax = growth_min_tax, growth_max_tax
    growth_fee_step = float(growth_auto_cfg.get("fee_step_pct", 0.35)) / 100.0
    growth_fee_step = float(np.clip(growth_fee_step, 0.0001, 0.02))
    growth_buy_push = float(np.clip(growth_auto_cfg.get("buy_push", 0.6), 0.0, 2.5))
    growth_buy_brake = float(np.clip(growth_auto_cfg.get("buy_brake", 0.5), 0.0, 2.5))
    growth_cash_buffer_pct = float(np.clip(growth_auto_cfg.get("cash_buffer_pct", 40.0), 0.0, 400.0))
    growth_surplus_spend_rate_pct = float(np.clip(growth_auto_cfg.get("surplus_spend_rate_pct", 20.0), 0.0, 100.0))
    growth_oracle_enabled = bool(growth_auto_cfg.get("oracle_enabled", False))
    growth_oracle_horizon = int(np.clip(growth_auto_cfg.get("oracle_horizon_steps", 6), 1, 48))
    growth_oracle_goal = str(growth_auto_cfg.get("target_metric", growth_auto_cfg.get("oracle_goal", algorithm_cfg.custom_params.get("primary_goal_mode", "market_cap"))))
    if growth_oracle_goal not in {"market_cap", "token_price", "hybrid"}:
        growth_oracle_goal = "market_cap"
    growth_oracle_weight = float(np.clip(growth_auto_cfg.get("oracle_weight", 1.0), 0.0, 10.0))
    growth_oracle_accuracy = float(np.clip(growth_auto_cfg.get("oracle_accuracy", 0.65), 0.0, 1.0))
    if growth_auto_enabled:
        algorithm_state.setdefault("growth_prev_cash", treasury_cash)
        seed_rate = float(np.clip(supply_plan.tx_tax_rate, growth_min_tax, growth_max_tax))
        algorithm_state.setdefault("growth_dynamic_tax_rate", seed_rate)
        algorithm_state.setdefault("growth_buy_bias", 0.0)
        # Set baseline LOWER so there's immediate surplus available
        # With 40% buffer, if treasury is 15k, we want threshold to be maybe 12k
        # So baseline should be 12k / 1.4 = 8571
        target_threshold = treasury_cash * 0.80  # Target threshold at 80% of initial cash
        initial_baseline = target_threshold / (1.0 + growth_cash_buffer_pct / 100.0)
        algorithm_state.setdefault("growth_baseline_cash", max(initial_baseline, 1.0))
        algorithm_state.setdefault("growth_surplus_cash", 0.0)
        if growth_oracle_enabled:
            algorithm_state.setdefault("growth_oracle_history", [])
            algorithm_state.setdefault("growth_oracle_bias", 0.0)
    else:
        algorithm_state.pop("growth_prev_cash", None)
        algorithm_state.pop("growth_dynamic_tax_rate", None)
        algorithm_state.pop("growth_buy_bias", None)
        algorithm_state.pop("growth_cash_delta", None)
        algorithm_state.pop("growth_baseline_cash", None)
        algorithm_state.pop("growth_surplus_cash", None)
        algorithm_state.pop("growth_surplus_spend_tokens", None)
        algorithm_state.pop("growth_oracle_history", None)
        algorithm_state.pop("growth_oracle_bias", None)
    crash_cfg = getattr(algorithm_cfg, "crash_defense", CrashDefenseSettings())
    crash_defense_enabled = bool(getattr(crash_cfg, "enabled", False))
    if crash_defense_enabled:
        algorithm_state.setdefault("crash_active_timer", 0)
        algorithm_state.setdefault("crash_cooldown_timer", 0)
        algorithm_state.setdefault("crash_budget_total", 0.0)
        algorithm_state.setdefault("crash_budget_remaining", 0.0)
        algorithm_state.setdefault("crash_price_window", [])
    else:
        algorithm_state.pop("crash_active_timer", None)
        algorithm_state.pop("crash_cooldown_timer", None)
        algorithm_state.pop("crash_budget_total", None)
        algorithm_state.pop("crash_budget_remaining", None)
        algorithm_state.pop("crash_price_window", None)
    price_window: List[float] = []

    policy_stage_active = False
    policy_stage_timer = 0
    policy_bootstrap_done = False
    policy_bankrupt = False
    policy_bankrupt_step: Optional[int] = None
    flow_memory = 0.0
    net_flow_memory = 0.0
    active_hype_events: List[Dict[str, float]] = []
    hype_cooldown = -hype_cfg.hype_cooldown_steps
    hype_exit_queue = 0.0

    for step in range(total_steps):
        prev_nav_price = nav_price
        peg_readiness = 0.0
        confidence_progress = 0.0
        prev_price = market_price
        time_years = step / sim.steps_per_year
        month = step + 1

        savings_pool *= 0.9
        growth_surplus_cash_step = 0.0
        growth_surplus_spend_tokens_step = 0.0
        growth_surplus_spend_cash_step = 0.0
        growth_fee_cash_step = 0.0
        growth_surplus_bucket = float(algorithm_state.get("growth_surplus_cash", 0.0)) if growth_auto_enabled else 0.0
        absolute_supply_adjustment_step = 0.0
        crash_detection = False
        crash_active_flag = False
        crash_cash_spend_step = 0.0
        crash_buy_tokens_step = 0.0
        crash_gas_spend_step = 0.0
        crash_drop_pct = 0.0
        crash_budget_remaining = float(algorithm_state.get("crash_budget_remaining", 0.0)) if crash_defense_enabled else 0.0
        if crash_defense_enabled:
            cooldown_timer = max(0, int(algorithm_state.get("crash_cooldown_timer", 0)))
            if cooldown_timer > 0:
                algorithm_state["crash_cooldown_timer"] = cooldown_timer - 1
            crash_active_timer = max(0, int(algorithm_state.get("crash_active_timer", 0)))
            crash_active_flag = crash_active_timer > 0
        else:
            crash_active_timer = 0

        if growth_auto_enabled:
            dynamic_tax_rate = float(
                np.clip(algorithm_state.get("growth_dynamic_tax_rate", supply_plan.tx_tax_rate), growth_min_tax, growth_max_tax)
            )
            growth_buy_bias = float(algorithm_state.get("growth_buy_bias", 0.0))
            price_basis_for_autopilot = prev_price
            if price_basis_for_autopilot <= epsilon * 10:
                price_basis_for_autopilot = max(fundamental_price, 0.05)
            price_basis_for_autopilot = float(max(price_basis_for_autopilot, 0.05))
            growth_oracle_bias = float(algorithm_state.get("growth_oracle_bias", 0.0)) if growth_oracle_enabled else 0.0
        else:
            dynamic_tax_rate = supply_plan.tx_tax_rate
            growth_buy_bias = 0.0
            price_basis_for_autopilot = float(max(prev_price, 0.05))
            growth_oracle_bias = 0.0
        tx_tax_rate = float(np.clip(dynamic_tax_rate, 0.0, 0.25))

        if algorithm_state.get("circuit_cooldown", 0) > 0:
            algorithm_state["circuit_cooldown"] = max(0, algorithm_state["circuit_cooldown"] - 1)
        algorithm_signal = 0.0
        algorithm_goal_progress = float(algorithm_state.get("goal_progress", 0.0))
        algo_flow_tokens = 0.0
        algo_executed = 0.0

        macro_growth_bonus = 0.0
        macro_churn_bonus = 0.0
        macro_confidence_shift = 0.0
        macro_flow_bias = 0.0
        liquidity_multiplier = 1.0
        capital_multiplier = 1.0

        if regime_cfg.enabled:
            regime_timer -= 1
            if regime_timer <= 0:
                regime_state, regime_timer, regime_intensity = sample_regime_state()
            steps_per_year = max(sim.steps_per_year, 1)
            innovation_decay_step = np.clip(regime_cfg.innovation_decay_rate, 0.0, 0.95) / steps_per_year
            innovation_level = max(0.0, innovation_level * (1 - innovation_decay_step))
            innovation_prob = 1 - (1 - np.clip(regime_cfg.innovation_chance, 0.0, 0.95)) ** (1 / steps_per_year)
            if rng.random() < innovation_prob:
                innovation_level += regime_cfg.innovation_capacity_boost * rng.uniform(0.5, 1.2)
            innovation_level = float(np.clip(innovation_level, 0.0, 1.6))
            drift_speed = regime_cfg.capacity_drift_speed / steps_per_year
            if regime_state == "expansion":
                capacity_shift += drift_speed * (0.6 + 0.4 * regime_intensity)
                macro_growth_bonus += regime_cfg.expansion_growth_bonus * regime_intensity
                macro_confidence_shift += regime_cfg.expansion_confidence_bonus * regime_intensity
                liquidity_multiplier *= 1 + regime_cfg.expansion_liquidity_bonus * regime_intensity
                capital_multiplier *= 1 + regime_cfg.capital_cycle_swing * 0.5 * regime_intensity
                macro_flow_bias = 0.04 * regime_intensity
            elif regime_state == "contraction":
                capacity_shift -= drift_speed * (0.8 + 0.2 * regime_intensity)
                macro_growth_bonus -= regime_cfg.contraction_growth_penalty * regime_intensity
                macro_churn_bonus += regime_cfg.contraction_churn_bonus * regime_intensity
                macro_confidence_shift -= regime_cfg.contraction_confidence_hit * regime_intensity
                liquidity_multiplier *= max(0.25, 1 - regime_cfg.contraction_liquidity_hit * regime_intensity)
                capital_multiplier *= max(0.25, 1 - regime_cfg.capital_cycle_swing * regime_intensity)
                macro_flow_bias = -0.05 * regime_intensity
            else:
                liquidity_multiplier *= max(0.6, 1 + rng.gauss(0.0, regime_cfg.capital_cycle_swing * 0.08))
                capital_multiplier *= max(0.5, 1 + rng.gauss(0.0, regime_cfg.capital_cycle_swing * 0.08))
            capacity_shift = float(np.clip(capacity_shift, regime_cfg.capacity_floor_shift, 1.0))

        effective_capacity_multiplier = 1.0 + capacity_shift + innovation_level
        effective_capacity = max(
            capacity_baseline * max(0.4, effective_capacity_multiplier),
            capacity_baseline * 0.6,
        )
        capacity = max(effective_capacity, float(sim.initial_users))
        confidence = float(np.clip(confidence + macro_confidence_shift, sim.confidence_floor, sim.confidence_ceiling))

        gold_noise = rng.gauss(0, step_vol_gold)
        gold_price *= math.exp(step_rate_gold + gold_noise - 0.5 * step_vol_gold ** 2)
        nav_price = gold_price

        deviation = (market_price - nav_price) / max(nav_price, epsilon)
        fundamental_gap = (market_price - fundamental_price) / max(fundamental_price, epsilon)

        if total_steps > 1:
            base_hold_progress = np.interp(step, [0, total_steps - 1], [sim.hold_ratio_start, sim.hold_ratio_end])
        else:
            base_hold_progress = sim.hold_ratio_end

        utilization = np.clip(users / capacity, 0.0, 2.0)
        saturation = max(0.0, 1.0 - utilization ** sim.adoption_saturation_power)

        price_momentum = np.tanh(((prev_price - demand_memory) / max(demand_memory, epsilon)) * sim.adoption_price_sensitivity)
        trend_growth = max(0.0, sim.base_user_growth + sim.growth_accel * time_years)
        hype_factor = max(0.0, hype_index)

        macro_penalty = 0.0
        macro_event = False
        shock_prob = max(0.0, sim.macro_shock_chance) / max(sim.steps_per_year, 1)
        if rng.random() < shock_prob:
            macro_penalty = rng.uniform(0.05, sim.macro_shock_magnitude)
            macro_event = True

        price_drawdown = max(0.0, (demand_memory - max(prev_price, epsilon)) / max(demand_memory, epsilon))
        saturation_gap = max(0.0, 1.0 - utilization)
        adoption_tailwind = (
            max(0.0, price_momentum) * 0.4
            + hype_index * 0.25
            + max(0.0, confidence - 0.55) * 0.35
        )
        adoption_drag = price_drawdown * 0.35 + macro_penalty * 0.6 + max(0.0, rng.gauss(0, sim.adoption_volatility * 0.4))
        early_adopter_gap = np.clip((2_500.0 - users) / 2_500.0, 0.0, 1.0)
        adoption_rate = sim.base_user_growth + macro_growth_bonus + adoption_tailwind - adoption_drag
        adoption_rate += early_adopter_gap * (0.02 + sim.base_user_growth * 0.6)
        adoption_rate = adoption_rate * (0.25 + 0.75 * saturation_gap)
        max_step_growth = min(sim.max_user_growth, 0.012)
        adoption_rate = float(np.clip(adoption_rate, sim.base_user_growth * 0.15, max_step_growth))

        churn_rate = sim.churn_rate + macro_churn_bonus
        churn_rate += max(0.0, -price_momentum) * sim.churn_price_sensitivity * 0.5
        churn_rate += price_drawdown * (0.35 + 0.4 * (1 - confidence))
        churn_rate += abs(price_momentum) * 0.1
        churn_rate += macro_penalty
        churn_rate -= confidence * 0.05
        churn_rate += max(0.0, rng.gauss(0, sim.adoption_volatility * 0.4))
        base_scale = 0.45 + 0.55 * np.clip(users / max(effective_capacity * 0.35, 1.0), 0.1, 1.0)
        churn_rate *= base_scale
        churn_rate = float(np.clip(churn_rate, 0.0, 0.12))

        if policy_bankrupt:
            adoption_rate *= 0.3
            churn_rate = min(0.75, churn_rate + 0.2)

        prev_users = users
        new_users = prev_users * adoption_rate
        raw_lost_users = prev_users * churn_rate
        sticky_churn = min(sticky_users, sticky_users * churn_rate * (0.1 + 0.2 * (1 - confidence)))
        flex_users = max(prev_users - sticky_users, 0.0)
        flex_churn = max(0.0, raw_lost_users - sticky_churn)
        max_structural_loss = max(prev_users - 50.0, 0.0) * 0.2
        lost_users = min(sticky_churn + flex_churn, max_structural_loss)
        users = max(50.0, prev_users + new_users - lost_users)
        sticky_users = max(0.0, sticky_users - min(sticky_churn, lost_users))
        sticky_gain = new_users * np.clip(0.22 + 0.35 * confidence, 0.05, 0.65)
        sticky_users = min(users, sticky_users + sticky_gain)

        hold_share = np.clip(
            base_hold_progress * (0.6 + 0.7 * confidence) * (0.65 + 0.4 * np.tanh(hype_index)),
            0.03,
            0.9,
        )
        base_liquidity = (sim.liquidity_base + sim.liquidity_per_user * users + free_float * 0.012) * liquidity_multiplier
        liquidity_depth = max(base_liquidity, 50_000.0)

        adoption_ratio = np.clip(users / max(effective_capacity, 1.0), 0.0, 1.5)
        liquidity_capacity = max(sim.liquidity_base * 3.0, 1.0)
        liquidity_ratio = np.clip(liquidity_depth / liquidity_capacity, 0.0, 2.5)
        confidence_span = max(sim.confidence_ceiling - sim.confidence_floor, epsilon)
        confidence_progress = np.clip((confidence - sim.confidence_floor) / confidence_span, 0.0, 1.0)

        retail_users = max(users * (1.0 - np.clip(sim.whale_user_fraction, 0.0, 0.6)), 0.0)
        whale_users = max(users - retail_users, 0.0)
        wallet_base = (
            sim.retail_wallet_mean
            * (0.7 + 0.6 * confidence)
            * (0.6 + 0.5 * np.tanh(hype_index - 0.3))
            * capital_multiplier
        )
        wallet_base *= math.exp(rng.gauss(-0.5 * sim.capital_noise ** 2, sim.capital_noise))
        wallet_base *= max(0.2, 1 + rng.gauss(0.0, sim.retail_wallet_spread * 0.35))
        wallet_base = float(np.clip(wallet_base, 120.0, 35_000.0))
        retail_capital = retail_users * wallet_base

        whale_wallet = wallet_base * max(2.0, sim.whale_balance_multiplier)
        whale_wallet *= (0.5 + 0.5 * confidence) * (0.7 + 0.5 * np.tanh(hype_index))
        whale_capital = whale_users * min(whale_wallet, wallet_base * 80.0)

        spec_activation = np.clip(
            0.45 * max(0.0, price_momentum)
            + 0.25 * np.tanh(hype_index)
            + 0.2 * max(0.0, -deviation)
            + 0.15 * confidence_progress,
            0.0,
            1.0,
        )
        spec_wallet = sim.speculator_base_balance * (0.8 + 0.6 * hype_index) * capital_multiplier
        speculative_capital = users * np.clip(sim.speculator_fraction, 0.0, 0.6) * spec_wallet * spec_activation

        hype_cooldown += 1
        event_buy_tokens = 0.0
        event_sell_tokens = 0.0
        event_new_users = 0.0
        event_whale_capital = 0.0
        surviving_events: List[Dict[str, Any]] = []
        for event in active_hype_events:
            if event.get("remaining", 0) <= 0:
                continue
            duration = max(event.get("duration", 1), 1)
            remaining = max(event.get("remaining", 0), 0)
            phase = remaining / duration
            intensity = event.get("intensity", 0.0) * (0.65 + 0.45 * phase)
            depth_reference = max(sim.liquidity_base, 80_000.0)
            buy_tokens = depth_reference * intensity / max(market_price, epsilon)
            buy_tokens *= 1.0 + np.clip(event.get("whale_bias", hype_cfg.whale_bias), 0.0, 3.5)
            buy_tokens = float(np.clip(buy_tokens, 0.0, free_float * 0.45))
            quick_flip = float(np.clip(event.get("quick_flip", hype_cfg.quick_flip_ratio), 0.0, 0.95))
            holder_ratio = float(np.clip(event.get("holder_ratio", hype_cfg.holder_ratio), 0.0, 0.9))
            quick_tokens = buy_tokens * quick_flip
            holder_tokens = buy_tokens * holder_ratio
            immediate_sell = quick_tokens * 0.35
            hype_exit_queue += max(0.0, quick_tokens - immediate_sell)
            event_buy_tokens += buy_tokens
            event_sell_tokens += immediate_sell
            event_new_users += max(40.0, users * 0.012) * intensity
            event_whale_capital += buy_tokens * market_price * 0.25 * np.clip(event.get("whale_bias", 0.0), 0.0, 3.0)
            event["remaining"] = remaining - 1
            event["age"] = event.get("age", 0) + 1
            if event["remaining"] > 0:
                surviving_events.append(event)
        active_hype_events = surviving_events

        exit_release = min(hype_exit_queue, hype_exit_queue * hype_cfg.retention_decay)
        hype_exit_queue = max(0.0, hype_exit_queue - exit_release)
        event_sell_tokens += exit_release

        if hype_cfg.enable_background_hype:
            base_prob = np.clip(hype_cfg.random_hype_chance, 0.0, 0.95)
            step_prob = 1 - (1 - base_prob) ** (1 / max(sim.steps_per_year, 1))
            heat_multiplier = 1 + 0.65 * max(0.0, price_momentum) + 0.35 * max(0.0, confidence - 0.45)
            if hype_cooldown >= 0 and rng.random() < min(0.95, step_prob * heat_multiplier):
                origin = "viral" if rng.random() < np.clip(hype_cfg.viral_spike_chance, 0.0, 0.75) else "organic"
                intensity = hype_cfg.base_intensity * rng.uniform(0.7, 1.4)
                if origin == "viral":
                    intensity *= hype_cfg.viral_intensity_multiplier * (1 + 0.2 * max(0.0, price_momentum))
                duration = rng.randint(max(1, hype_cfg.min_duration_steps), max(hype_cfg.min_duration_steps, hype_cfg.max_duration_steps))
                if origin == "viral":
                    duration = int(duration * 1.5)
                active_hype_events.append(
                    {
                        "intensity": float(max(0.05, intensity)),
                        "duration": int(max(1, duration)),
                        "remaining": int(max(1, duration)),
                        "origin": origin,
                        "whale_bias": float(np.clip(hype_cfg.whale_bias * rng.uniform(0.7, 1.4), 0.0, 3.0)),
                        "quick_flip": float(np.clip(hype_cfg.quick_flip_ratio * rng.uniform(0.6, 1.5), 0.05, 0.95)),
                        "holder_ratio": float(np.clip(hype_cfg.holder_ratio * rng.uniform(0.7, 1.3), 0.0, 0.9)),
                        "age": 0,
                    }
                )
                hype_cooldown = -hype_cfg.hype_cooldown_steps

        if event_new_users > 0:
            users = min(effective_capacity * 1.2, users + event_new_users)
            liquidity_depth += event_new_users * sim.liquidity_per_user * 0.25
            liquidity_depth = max(liquidity_depth, sim.liquidity_base * 0.75)
            sticky_users += event_new_users * np.clip(0.18 + 0.3 * confidence, 0.05, 0.6)
            sticky_users = min(sticky_users, users)
        whale_capital += event_whale_capital

        community_capital = max(0.0, retail_capital + whale_capital + speculative_capital)
        capital_price = community_capital / max(free_float, epsilon)
        capital_price = float(np.clip(capital_price, sim.intrinsic_value_start * 0.4, nav_price * 1.2))

        readiness_input = (
            (adoption_ratio ** max(sim.intrinsic_growth_exponent, 0.1)) * 0.55
            + (liquidity_ratio ** 0.5) * sim.intrinsic_liquidity_weight * 0.25
            + confidence_progress * sim.intrinsic_confidence_weight * 0.35
        )
        peg_readiness = float(np.clip(np.tanh(readiness_input), 0.0, 1.0))

        intrinsic_floor = max(sim.intrinsic_value_start, epsilon)
        intrinsic_ceiling = nav_price * (0.75 + 0.2 * confidence_progress)
        capital_anchor = min(capital_price, intrinsic_ceiling)
        target_fundamental = intrinsic_floor * (1.0 - peg_readiness) + capital_anchor * peg_readiness
        intrinsic_shock = math.exp(rng.gauss(0.0, sim.intrinsic_noise))
        target_fundamental = max(intrinsic_floor * 0.6, target_fundamental * intrinsic_shock)
        fundamental_price += (target_fundamental - fundamental_price) * 0.32
        fundamental_price = float(np.clip(fundamental_price, intrinsic_floor * 0.55, nav_price * 1.05))
        target_hold_value = community_capital * hold_share
        adjust_rate = 0.18 + 0.45 * adoption_rate
        desired_user_tokens = target_hold_value / max(market_price, epsilon)
        desired_user_tokens = float(np.clip(desired_user_tokens, 0.0, free_float * 0.95))
        user_flow = (desired_user_tokens - user_tokens) * adjust_rate
        sentiment_flow = community_capital / max(market_price, epsilon) * (confidence - 0.5) * 0.04
        hype_flow = community_capital / max(market_price, epsilon) * np.tanh(hype_index - 0.45) * 0.03
        noise_flow = rng.gauss(0, sim.organic_noise * max(1.0, free_float / 20_000_000.0) ** 0.25)

        organic_flow = user_flow + sentiment_flow + hype_flow + noise_flow
        organic_flow += event_buy_tokens - event_sell_tokens
        organic_flow += -fundamental_gap * sim.arbitrage_flow_strength * (free_float / max(market_price, epsilon) * 0.015)
        if macro_flow_bias != 0.0:
            organic_flow += macro_flow_bias * free_float * 0.01
        bootstrap_flow = 0.0
        bootstrap_window = max(int(sim.steps_per_year * 2.5), 1)
        if step < bootstrap_window:
            price_gap = np.clip((nav_price - market_price) / max(nav_price, epsilon), 0.0, 1.5)
            bootstrap_intensity = (1.0 - peg_readiness) * 0.5 + confidence_progress * 0.3
            bootstrap_value = community_capital * 0.12 * bootstrap_intensity
            bootstrap_flow = min(free_float * 0.08, bootstrap_value / max(market_price, epsilon)) * price_gap
            organic_flow += bootstrap_flow
        max_flow = max(free_float * 0.12, 5_000.0)
        organic_flow = float(np.clip(organic_flow, -max_flow, max_flow))
        organic_flow = 0.7 * organic_flow + 0.3 * flow_memory
        flow_memory = organic_flow
        fundamental_gap = (market_price - fundamental_price) / max(fundamental_price, epsilon)

        price_window.append(market_price)
        if len(price_window) > sim.steps_per_year:
            price_window.pop(0)
        if crash_defense_enabled:
            crash_price_window = list(algorithm_state.get("crash_price_window", []))
            crash_price_window.append(market_price)
            max_len = max(32, int(crash_cfg.detection_window) * 4)
            if len(crash_price_window) > max_len:
                crash_price_window = crash_price_window[-max_len:]
            algorithm_state["crash_price_window"] = crash_price_window
            window = max(2, int(crash_cfg.detection_window))
            if len(crash_price_window) >= window:
                recent_slice = crash_price_window[-window:]
                peak_price = max(recent_slice)
                if peak_price > epsilon:
                    crash_drop_pct = max(0.0, (peak_price - market_price) / peak_price * 100.0)
                else:
                    crash_drop_pct = 0.0
                if crash_drop_pct >= crash_cfg.drop_threshold_pct and algorithm_state.get("crash_cooldown_timer", 0) <= 0:
                    crash_detection = True
                    crash_active_flag = True
                    crash_active_timer = max(int(crash_cfg.stabilize_steps), 1)
                    algorithm_state["crash_active_timer"] = crash_active_timer
                    algorithm_state["crash_cooldown_timer"] = max(int(crash_cfg.cooldown_steps), 1)
                    budget_total = treasury_cash * max(crash_cfg.resource_commitment_pct, 0.0) / 100.0
                    algorithm_state["crash_budget_total"] = float(budget_total)
                    algorithm_state["crash_budget_remaining"] = float(budget_total)
                    crash_budget_remaining = float(budget_total)
                    algorithm_state["circuit_cooldown"] = max(
                        algorithm_state.get("circuit_cooldown", 0), int(crash_cfg.circuit_lock_steps)
                    )
            else:
                crash_drop_pct = 0.0
            crash_active_timer = max(0, int(algorithm_state.get("crash_active_timer", 0)))
        if len(price_window) >= 2:
            velocity = (price_window[-1] - price_window[0]) / max(price_window[0], epsilon)
        else:
            velocity = 0.0
        if len(price_window) >= 4:
            rel_changes = np.diff(price_window) / np.clip(price_window[:-1], epsilon, None)
            realized_vol = float(np.std(rel_changes))
        else:
            realized_vol = 0.0

        algo_metrics = {
            "price": market_price,
            "deviation": deviation,
            "growth": adoption_rate,
            "volatility": realized_vol,
            "treasury_nav_gap": (nav_price - market_price) / max(nav_price, epsilon),
            "velocity": velocity,
            "confidence": confidence,
            "users": users,
        }
        algo_metrics["custom_index"] = (
            0.5 * algo_metrics["deviation"] - 0.3 * algo_metrics["volatility"] + 0.2 * algo_metrics["velocity"]
        )

        if algorithm_cfg.enabled or algorithm_cfg.master_switch:
            signal_sum = 0.0
            weight_sum = 0.0
            satisfaction_sum = 0.0
            for objective in algorithm_cfg.objectives:
                if not getattr(objective, "enabled", True):
                    continue
                signal, satisfaction = _compute_objective_signal(objective, algo_metrics)
                signal_sum += signal
                weight = abs(objective.weight) if objective.weight else 1.0
                weight_sum += weight
                satisfaction_sum += satisfaction * weight
            if weight_sum > 0:
                algorithm_signal = float(np.clip(signal_sum / weight_sum, -5.0, 5.0))
                algorithm_goal_progress = float(np.clip(satisfaction_sum / weight_sum, 0.0, 1.5))
            algorithm_state["goal_progress"] = algorithm_goal_progress
            algorithm_state["last_signal"] = algorithm_signal

        hype_decay = 1 - sim.hype_decay
        hype_noise = rng.gauss(0.0, 0.015 + sim.organic_noise * 0.25)
        event_signal = 0.0
        if free_float > 0:
            event_signal = np.clip((event_buy_tokens - event_sell_tokens) / max(free_float, epsilon), -0.6, 1.4)
        hype_growth_input = (
            sim.hype_sensitivity_growth * adoption_rate
            + sim.hype_sensitivity_price * max(0.0, price_momentum)
            + event_signal
        )
        if growth_auto_enabled:
            hype_growth_input += algorithm_state.pop("oracle_hype_boost", 0.0)
        hype_index = max(0.0, hype_index * hype_decay + hype_growth_input + hype_noise)
        hype_index = float(np.clip(hype_index, 0.0, 4.5))
        demand_memory = 0.6 * prev_price + 0.4 * demand_memory

        total_new_users = max(0.0, new_users + event_new_users)
        price_ratio_to_goal = market_price / max(supply_plan.goal_price, epsilon)
        if market_price >= supply_plan.goal_price:
            last_goal_hit_step = step
            target_emission_rate = max(
                target_emission_rate,
                supply_plan.goal_emission_per_user * max(1.0, supply_plan.goal_release_multiplier),
            )
        if last_goal_hit_step is not None and step >= next_halving_step:
            target_emission_rate = max(
                supply_plan.baseline_emission_per_user,
                target_emission_rate * np.clip(supply_plan.halving_factor, 0.1, 1.0),
            )
            next_halving_step += max(supply_plan.halving_interval_steps, 1)

        reward_gate = 0.0
        if last_goal_hit_step is not None:
            reward_gate = np.clip(price_ratio_to_goal * supply_plan.goal_release_multiplier, 0.0, 1.0)
        desired_emission_rate = supply_plan.baseline_emission_per_user + (
            (target_emission_rate - supply_plan.baseline_emission_per_user) * reward_gate
        )
        emission_rate += (desired_emission_rate - emission_rate) * 0.35
        emission_rate = max(0.0, emission_rate)

        minted_step = max(0.0, total_new_users * emission_rate)
        if policy.module_mint_control and policy.enabled:
            peg_relative = (market_price - policy.mint_support_floor) / max(
                policy.mint_support_ceiling - policy.mint_support_floor, epsilon
            )
            peg_relative = np.clip(peg_relative, -2.0, 2.0)
            minted_step *= np.clip(1 + policy.mint_ramp_strength * peg_relative, 0.0, 2.75)
        if (algorithm_cfg.enabled or algorithm_cfg.master_switch) and algo_modules.adaptive_mint and algorithm_signal < 0:
            minted_step *= 1 + min(0.8, abs(algorithm_signal))
        available_supply = max(0.0, supply_plan.supply_hard_cap - total_supply)
        regime = supply_plan.supply_regime
        if regime == "decay":
            half_life_steps = max(1, int(supply_plan.decay_half_life_years * sim.steps_per_year))
            decay_multiplier = 0.5 ** (step / half_life_steps)
            minted_step *= float(np.clip(decay_multiplier, 0.0, 1.0))
        elif regime == "adaptive":
            span = max(supply_plan.adaptive_ceiling_price - supply_plan.adaptive_floor_price, epsilon)
            relative = (market_price - supply_plan.adaptive_floor_price) / span
            relative = float(np.clip(relative, -1.5, 2.0))
            if market_price < supply_plan.adaptive_floor_price:
                minted_step *= 0.15
            minted_step *= np.clip(relative + 0.6, 0.0, 2.0)
        elif regime == "hard_cap":
            minted_step = min(minted_step, available_supply)
        minted_step = min(minted_step, available_supply)

        mint_scale = float(np.clip(algorithm_state.pop("oracle_mint_bias", 1.0), 0.2, 2.5)) if growth_auto_enabled else 1.0
        minted_step *= mint_scale

        guard_level = supply_plan.inflation_guard_price * nav_price
        if market_price < guard_level:
            inflation_guard_timer = max(inflation_guard_timer, supply_plan.inflation_guard_cooldown)
        if inflation_guard_timer > 0:
            minted_step *= 0.1
            inflation_guard_timer = max(0, inflation_guard_timer - 1)

        release_bias = np.clip(0.35 + 0.25 * np.tanh(price_ratio_to_goal), 0.15, 0.9)
        if last_goal_hit_step is not None:
            release_bias = min(0.95, release_bias + 0.08 * supply_plan.goal_release_multiplier)
        mint_to_market = minted_step * release_bias
        mint_to_treasury = minted_step - mint_to_market
        free_float += mint_to_market
        treasury_tokens += mint_to_treasury
        minted_cumulative += minted_step

        unlock_gate = np.clip(price_ratio_to_goal + confidence_progress * 0.35, 0.0, 1.5)
        unlock_noise = rng.uniform(1 - supply_plan.unlock_jitter, 1 + supply_plan.unlock_jitter)
        unlock_tokens = min(
            locked_supply,
            locked_supply * supply_plan.unlock_slope * unlock_gate * max(0.35, unlock_noise),
        )
        if unlock_tokens > 0:
            locked_supply = max(0.0, locked_supply - unlock_tokens)
            free_float += unlock_tokens
        if supply_plan.price_trigger_unlock and market_price >= supply_plan.price_trigger_unlock:
            trigger_unlock = min(
                locked_supply,
                locked_supply * supply_plan.price_trigger_unlock_fraction,
            )
            if trigger_unlock > 0:
                locked_supply = max(0.0, locked_supply - trigger_unlock)
                free_float += trigger_unlock
                unlock_tokens += trigger_unlock
        burn_vault_release_step = 0.0

        burned_step = 0.0
        if supply_plan.price_trigger_burn and market_price >= supply_plan.price_trigger_burn:
            burn_cap = max(0.0, free_float - max(supply_plan.burn_floor_supply, sim.supply_floor))
            burn_tokens = min(burn_cap, free_float * supply_plan.price_trigger_burn_fraction)
            if burn_tokens > 0:
                free_float -= burn_tokens
                burned_step += burn_tokens
                burned_cumulative += burn_tokens
        if (algorithm_cfg.enabled or algorithm_cfg.master_switch) and algo_modules.adaptive_burn and algorithm_signal > 0:
            adaptive_burn = min(
                max(0.0, free_float - max(supply_plan.burn_floor_supply, sim.supply_floor)),
                free_float * 0.02 * np.clip(algorithm_signal, 0.0, 3.0),
            )
            if adaptive_burn > 0:
                free_float -= adaptive_burn
                burned_step += adaptive_burn
                burned_cumulative += adaptive_burn
        if supply_plan.supply_reversal_price and market_price <= supply_plan.supply_reversal_price:
            relock_cap = max(0.0, free_float - sim.supply_floor)
            relock_tokens = min(relock_cap, free_float * supply_plan.supply_reversal_fraction)
            if relock_tokens > 0:
                free_float -= relock_tokens
                locked_supply += relock_tokens

        stage_ready = True
        if policy.activation_step > 0 and step < policy.activation_step:
            stage_ready = False
        if policy.activation_price > 0 and market_price < policy.activation_price:
            stage_ready = False
        if policy.activation_confidence > 0 and confidence < policy.activation_confidence:
            stage_ready = False

        if policy_stage_active:
            policy_stage_timer += 1
        if not policy_stage_active and stage_ready and policy.enabled:
            policy_stage_active = True
            policy_stage_timer = 1
            if not policy_bootstrap_done:
                treasury_cash += policy.bootstrap_cash
                treasury_tokens += policy.bootstrap_tokens
                policy_bootstrap_done = True
        elif not policy_stage_active:
            policy_stage_timer = 0

        if policy_stage_active and policy.ramp_up_steps > 0:
            stage_multiplier = min(1.0, policy_stage_timer / float(max(1, policy.ramp_up_steps)))
        elif policy_stage_active:
            stage_multiplier = 1.0
        else:
            stage_multiplier = 0.0

        central_bank_live = policy.enabled and policy_stage_active and not policy_bankrupt

        available_float_tokens = max(free_float, 1.0)
        book_tokens_capacity = max(liquidity_depth / max(market_price, epsilon), 1.0)
        attack_token_cap = max(available_float_tokens * 0.5, 1.0)

        active_attacks = [a for a in attack_events if a.start_step <= step < a.start_step + a.duration]
        attack_flow_tokens = 0.0
        liquidity_penalty = 0.0
        for attack in active_attacks:
            depth_reference = max(liquidity_depth, 100_000.0)
            size_tokens = attack.magnitude * depth_reference / max(market_price, epsilon)
            jitter = rng.uniform(0.6, 1.4)
            signed_tokens = 0.0
            if attack.side == "buy":
                signed_tokens = size_tokens * jitter
            elif attack.side == "sell":
                signed_tokens = -size_tokens * jitter
            else:
                direction = rng.choice([-1.0, 1.0])
                signed_tokens = size_tokens * jitter * direction
            signed_tokens = float(np.clip(signed_tokens, -attack_token_cap, attack_token_cap))
            if abs(signed_tokens) > epsilon:
                trade_rows.append(
                    {
                        "step": step,
                        "action": "buy" if signed_tokens > 0 else "sell",
                        "tokens": abs(signed_tokens),
                        "price": market_price,
                        "name": attack.name,
                        "tag": f"scripted_{attack.side}",
                    }
                )
            attack_flow_tokens += signed_tokens
            liquidity_penalty = max(liquidity_penalty, attack.liquidity_drop)

        auto_flow_tokens = 0.0
        auto_mode = "idle"
        auto_signal = 0.0
        auto_meta_rows: List[Dict[str, Any]] = []
        phase_counter: Counter = Counter()
        intent_samples: List[float] = []

        if attacker_settings.auto_enabled:
            total_steps = max(total_steps, 1)
            time_elapsed = step / max(total_steps - 1, 1)
            time_remaining = max(0.0, 1.0 - time_elapsed)
            for profile, state in zip(profile_bundle, attacker_states):
                context = {
                    "price": market_price,
                    "nav_price": nav_price,
                    "fundamental_price": fundamental_price,
                    "deviation": deviation,
                    "momentum": price_momentum,
                    "time_remaining": time_remaining,
                    "time_elapsed": time_elapsed,
                    "step": step,
                    "total_steps": total_steps,
                    "liquidity_depth": liquidity_depth,
                    "free_float": free_float,
                    "confidence": confidence,
                }
                flow, penalty, phase, meta = _attacker_phase_step(
                    rng, attacker_settings, profile, state, context
                )
                if abs(flow) > epsilon:
                    trade_rows.append(
                        {
                            "step": step,
                            "action": "buy" if flow > 0 else "sell",
                            "tokens": abs(flow),
                            "price": market_price,
                            "name": state.label,
                            "tag": meta.get("action", "auto"),
                        }
                    )
                auto_flow_tokens += flow
                liquidity_penalty = max(liquidity_penalty, penalty)
                phase_counter[phase] += 1
                intent_samples.append(meta.get("intent", 0.0))
                auto_meta_rows.append(meta)
                if attacker_settings.objective == "maximize_cash":
                    target_cash = profile.capital * (1 + attacker_settings.final_push_aggression)
                    state.objective_progress = state.cash / max(target_cash, 1.0)
                elif attacker_settings.objective == "maximize_tokens":
                    state.objective_progress = state.tokens / max(attack_token_cap, 1.0)
                elif attacker_settings.objective == "maximize_pnl":
                    state.objective_progress = state.pnl / max(profile.capital, 1.0)
                elif attacker_settings.objective == "destabilize":
                    state.objective_progress = abs(deviation)
                else:
                    total_value = state.cash + state.tokens * market_price
                    state.objective_progress = total_value / max(profile.capital, 1.0)

            if intent_samples:
                auto_signal = float(np.clip(np.mean(intent_samples), -3.0, 3.0))
            auto_mode = phase_counter.most_common(1)[0][0] if phase_counter else "idle"

        attack_flow_tokens += auto_flow_tokens
        liquidity_depth = max(liquidity_depth * (1 - liquidity_penalty), 40_000.0)
        effective_depth = liquidity_depth
        algorithm_active = algorithm_cfg.enabled or algorithm_cfg.master_switch
        algo_exec_enabled = algorithm_active or growth_auto_enabled
        algo_modules = algorithm_cfg.modules
        lever_defaults = {"buy": True, "sell": True, "mint_adjust": True, "burn_adjust": True, "hype_boost": True}
        lever_controls = dict(getattr(algorithm_cfg, "leverage_controls", lever_defaults))
        if algorithm_active:
            if algo_modules.liquidity_support:
                support_strength = 0.35
                calm_factor = max(0.0, 1 - abs(deviation) / max(algo_modules.circuit_threshold or 0.05, 0.05))
                effective_depth *= 1 + support_strength * calm_factor
            if algo_modules.circuit_breaker and abs(deviation) >= max(algo_modules.circuit_threshold, 1e-4):
                algorithm_state["circuit_cooldown"] = max(
                    algorithm_state.get("circuit_cooldown", 0), algo_modules.circuit_cooldown_steps
                )
            if algorithm_state.get("circuit_cooldown", 0) > 0:
                damp_factor = 1 + 0.5 * algorithm_state["circuit_cooldown"]
                organic_flow /= damp_factor
                liquidity_penalty = max(liquidity_penalty, min(0.6, algo_modules.circuit_threshold * 0.6))
            if algo_modules.trend_follow:
                algo_flow_tokens += np.tanh(price_momentum * 2.0) * (effective_depth / max(nav_price, epsilon)) * 0.08
            if algo_modules.drawdown_guard and price_drawdown > algo_modules.drawdown_threshold:
                guard = (price_drawdown - algo_modules.drawdown_threshold) * (effective_depth / max(nav_price, epsilon)) * 0.18
                algo_flow_tokens += guard
            floor_price = algo_modules.floor_price
            ceiling_price = algo_modules.ceiling_price
            if floor_price is not None and market_price < floor_price:
                diff = floor_price - market_price
                algo_flow_tokens += diff / max(market_price, epsilon) * (effective_depth / max(nav_price, epsilon)) * 0.25
            if ceiling_price is not None and market_price > ceiling_price:
                diff = market_price - ceiling_price
                algo_flow_tokens -= diff / max(market_price, epsilon) * (effective_depth / max(nav_price, epsilon)) * 0.25
            if algo_modules.objective_lock_when_met and algorithm_goal_progress >= 0.95:
                algorithm_signal *= 0.25
            algo_flow_tokens += algorithm_signal * (effective_depth / max(nav_price, epsilon)) * 0.12
            if algo_modules.velocity_targets:
                algo_flow_tokens -= velocity * (effective_depth / max(nav_price, epsilon)) * 0.05

        if growth_auto_enabled:
            combined_bias = growth_buy_bias
            if combined_bias != 0.0:
                algo_flow_tokens += combined_bias * (effective_depth / max(nav_price, epsilon))

        oracle_spend_multiplier = 1.0
        oracle_fee_adjust = 0.0
        oracle_bias_delta = 0.0
        oracle_mint_scale = 1.0
        oracle_hype_boost = 0.0

        if growth_auto_enabled:
            baseline_cash = float(max(algorithm_state.get("growth_baseline_cash", treasury_cash), 1.0))
            approx_circulating = free_float + treasury_tokens * (0.08 + 0.22 * confidence)
            approx_circulating = max(approx_circulating, 1.0)
            approx_cap = market_price * approx_circulating
            prev_cap = float(algorithm_state.get("oracle_prev_cap", approx_cap))
            prev_price_ref = float(algorithm_state.get("oracle_prev_price", market_price))
            if growth_oracle_goal == "market_cap":
                focus_metric = approx_cap
            elif growth_oracle_goal == "token_price":
                focus_metric = market_price
            else:
                focus_metric = 0.5 * (
                    (market_price / max(prev_price_ref, epsilon))
                    + (approx_cap / max(prev_cap, epsilon))
                )

            if growth_oracle_enabled and growth_oracle_accuracy > 0.0:
                fee_candidates = (-1.0, 0.0, 1.0)
                bias_candidates = (-0.7, 0.0, 0.9)
                candidates: List[Dict[str, float]] = []
                cash_pressure = (treasury_cash - baseline_cash) / max(baseline_cash, 1.0)
                hype_space = max(0.0, 1.5 - hype_index)
                for fee_sign in fee_candidates:
                    fee_delta = fee_sign * growth_fee_step
                    for bias_ratio in bias_candidates:
                        if bias_ratio > 0 and not lever_controls.get("buy", True):
                            continue
                        if bias_ratio < 0 and not lever_controls.get("sell", True):
                            continue
                        spend_mult = 1.0 + bias_ratio * 0.4
                        mint_delta = 0.15 * bias_ratio if lever_controls.get("mint_adjust", True) else 0.0
                        burn_delta = 0.12 * (-bias_ratio) if lever_controls.get("burn_adjust", True) else 0.0
                        hype_bonus = max(0.0, bias_ratio) * 0.15 if lever_controls.get("hype_boost", True) else 0.0

                        momentum_term = price_momentum * (bias_ratio * 0.9 + (spend_mult - 1.0))
                        hype_term = hype_space * max(0.0, bias_ratio) * 0.35 - max(0.0, hype_index - 1.5) * abs(fee_delta) * 4.0
                        cash_term = -fee_delta * 95.0 * cash_pressure
                        deviation_term = -abs(deviation) * abs(fee_delta) * 140.0
                        burn_term = burn_delta * 40.0
                        candidate_score = focus_metric + momentum_term + hype_term + cash_term + deviation_term + burn_term
                        if growth_oracle_goal == "hybrid":
                            candidate_score += (momentum_term * 0.45 + hype_term * 0.25)
                        candidates.append(
                            {
                                "score": candidate_score,
                                "fee_delta": fee_delta,
                                "bias_ratio": bias_ratio,
                                "spend_mult": spend_mult,
                                "mint_delta": mint_delta,
                                "hype_boost": hype_bonus,
                            }
                        )
                if candidates:
                    best = max(candidates, key=lambda item: item["score"])
                    blend = growth_oracle_accuracy
                    oracle_fee_adjust = best["fee_delta"] * blend
                    bias_delta_raw = best["bias_ratio"] * growth_buy_push * blend
                    if bias_delta_raw > 0 and not lever_controls.get("buy", True):
                        bias_delta_raw = 0.0
                    if bias_delta_raw < 0 and not lever_controls.get("sell", True):
                        bias_delta_raw = 0.0
                    oracle_bias_delta = bias_delta_raw
                    oracle_spend_multiplier = 1.0 + (best["spend_mult"] - 1.0) * blend
                    if lever_controls.get("mint_adjust", True):
                        oracle_mint_scale = float(np.clip(1.0 + best["mint_delta"] * blend, 0.4, 1.9))
                    if lever_controls.get("hype_boost", True):
                        oracle_hype_boost = best["hype_boost"] * blend

        if growth_auto_enabled:
            baseline_cash = float(max(algorithm_state.get("growth_baseline_cash", treasury_cash), 1.0))
            buffer_multiplier = 1.0 + growth_cash_buffer_pct / 100.0
            buffer_threshold = baseline_cash * buffer_multiplier
            surplus_bucket = float(max(algorithm_state.get("growth_surplus_cash", 0.0), 0.0))
            new_surplus_cash = max(0.0, treasury_cash - buffer_threshold)
            if new_surplus_cash > 0:
                surplus_bucket += new_surplus_cash
                growth_surplus_cash_step = new_surplus_cash
            else:
                growth_surplus_cash_step = 0.0
            spend_fraction = np.clip(growth_surplus_spend_rate_pct / 100.0, 0.0, 1.0)
            oracle_spend_multiplier = 1.0
            if growth_oracle_enabled:
                oracle_spend_multiplier += np.clip(growth_oracle_bias, -1.5, 2.0) * 0.35
                oracle_spend_multiplier = max(0.0, oracle_spend_multiplier)
            oracle_spend_multiplier = float(np.clip(oracle_spend_multiplier, 0.2, 2.5))
            max_spend_this_step = surplus_bucket * spend_fraction * oracle_spend_multiplier
            # Allow spending directly from treasury cash above baseline, not just from bucket
            available_cash = max(0.0, treasury_cash - baseline_cash * 1.05)  # Keep 5% safety margin
            spend_cash = min(max_spend_this_step, available_cash)
            if spend_cash > epsilon:
                growth_surplus_spend_tokens_step = spend_cash / price_basis_for_autopilot
                algo_flow_tokens += growth_surplus_spend_tokens_step
                surplus_bucket = max(0.0, surplus_bucket - spend_cash)
                growth_surplus_spend_cash_step = spend_cash
            else:
                growth_surplus_spend_tokens_step = 0.0
            algorithm_state["growth_surplus_cash"] = float(surplus_bucket)
            growth_surplus_bucket = float(surplus_bucket)
            # Only drift baseline upward slowly, and never let it exceed a safe threshold
            if treasury_cash >= baseline_cash * 1.2:  # Only drift if we have comfortable margin
                drift_alpha = 0.02  # Very slow drift upward
                drift_target = baseline_cash + (treasury_cash - baseline_cash) * drift_alpha
            else:
                # Don't drift if treasury is close to baseline - keep baseline stable
                drift_target = baseline_cash
            algorithm_state["growth_baseline_cash"] = float(max(drift_target, 1.0))
            algorithm_state["growth_surplus_spend_tokens"] = float(growth_surplus_spend_tokens_step)

        financing_cash = 0.0
        if policy.financing_contrib_rate > 0 and (policy.financing_pre_stage or central_bank_live):
            financing_scale = stage_multiplier if central_bank_live else 1.0
            skim_base = abs(organic_flow) + max(0.0, mint_to_market)
            financing_cash = policy.financing_contrib_rate * skim_base * market_price * financing_scale
            if financing_cash > 0:
                treasury_cash += financing_cash
        
        # Growth autopilot fee capture (overrides or adds to normal financing)
        growth_financing_cash = 0.0
        if growth_auto_enabled and growth_fee_capture_rate > 0:
            skim_base = abs(organic_flow) + max(0.0, mint_to_market)
            growth_financing_cash = growth_fee_capture_rate * skim_base * price_basis_for_autopilot
            if growth_financing_cash > 0:
                if growth_override_financing:
                    # Replace financing_cash with growth version
                    treasury_cash -= financing_cash  # Remove original
                    treasury_cash += growth_financing_cash  # Add growth version
                    financing_cash = growth_financing_cash
                else:
                    # Add growth financing on top of normal financing
                    treasury_cash += growth_financing_cash
                    financing_cash += growth_financing_cash
                growth_fee_cash_step = growth_financing_cash

        policy_flow = 0.0
        savings_tokens = 0.0
        subsidy = 0.0
        abs_dev = abs(deviation)
        fee_bias = 1.0
        soft_band = max(policy.nav_band_soft, 1e-4)
        hard_band = max(policy.nav_band_hard, soft_band)

        if central_bank_live and policy.module_liquidity_support and policy.liquidity_support_strength > 0:
            band_ref = max(hard_band, 1e-6)
            support_ratio = max(0.0, 1 - min(abs_dev, band_ref) / band_ref)
            support_boost = policy.liquidity_support_strength * stage_multiplier * support_ratio
            effective_depth *= 1 + support_boost

        if central_bank_live and policy.module_fee_incentives:
            bias = np.clip(abs_dev / soft_band, 0.0, 3.0)
            if abs_dev <= soft_band:
                fee_bias -= policy.fee_rebate_strength * stage_multiplier * (1 - bias)
            else:
                span = max(hard_band - soft_band, 1e-6)
                fee_bias += policy.fee_penalty_strength * stage_multiplier * ((abs_dev - soft_band) / span)

        if central_bank_live and policy.module_circuit_breaker and abs_dev >= policy.breaker_threshold:
            fee_bias *= 1 + policy.breaker_flow_shock * stage_multiplier

        fee_bias = float(np.clip(fee_bias, 0.3, 5.0))
        if fee_bias != 1.0:
            organic_flow /= fee_bias

        if central_bank_live and policy.module_savings and deviation < 0:
            savings_scale = min(abs_dev / soft_band, 1.5)
            savings_tokens = policy.savings_strength * stage_multiplier * savings_scale * free_float * 0.05
            organic_flow += savings_tokens

        if central_bank_live and policy.module_policy_arbitrage:
            policy_flow += -deviation * policy.arb_flow_bonus * stage_multiplier * (free_float / max(market_price, epsilon)) * 0.1

        if central_bank_live and policy.module_omo:
            max_tokens_to_sell = max(0.0, policy.max_omo_fraction * treasury_tokens)
            max_tokens_to_buy = max(0.0, policy.max_omo_fraction * (treasury_cash / max(nav_price, epsilon)))
            desired_tokens = policy.omo_strength * stage_multiplier * deviation * (effective_depth / max(nav_price, epsilon))

            if deviation > soft_band and max_tokens_to_sell > 0:
                sell_tokens = min(max_tokens_to_sell, abs(desired_tokens))
                policy_flow -= sell_tokens
                treasury_tokens -= sell_tokens
                treasury_cash += sell_tokens * nav_price
                free_float += sell_tokens
            elif deviation < -soft_band and max_tokens_to_buy > 0:
                buy_tokens = min(max_tokens_to_buy, abs(desired_tokens))
                policy_flow += buy_tokens
                treasury_tokens += buy_tokens
                treasury_cash -= buy_tokens * nav_price
        prev_flow_memory = float(algorithm_state.get("algo_flow_memory", 0.0))
        smoothed_flow = 0.65 * algo_flow_tokens + 0.35 * prev_flow_memory
        max_depth_tokens = (effective_depth / max(nav_price, epsilon)) * 0.6
        max_cash_tokens = (treasury_cash / max(market_price, epsilon)) * 0.6
        max_flow_tokens = max(max_depth_tokens, max_cash_tokens, 1.0)
        smoothed_flow = float(np.clip(smoothed_flow, -max_flow_tokens, max_flow_tokens))
        algorithm_state["algo_flow_memory"] = smoothed_flow
        algo_flow_tokens = smoothed_flow

        if algo_exec_enabled and abs(algo_flow_tokens) > epsilon:
            if algo_flow_tokens > 0:
                max_tokens_to_buy = treasury_cash / max(market_price, epsilon)
                buy_tokens = min(algo_flow_tokens, max_tokens_to_buy)
                if buy_tokens > 0:
                    policy_flow += buy_tokens
                    treasury_tokens += buy_tokens
                    treasury_cash -= buy_tokens * market_price
                    algo_executed = buy_tokens
            else:
                sell_tokens = min(abs(algo_flow_tokens), treasury_tokens)
                if sell_tokens > 0:
                    policy_flow -= sell_tokens
                    treasury_tokens -= sell_tokens
                    treasury_cash += sell_tokens * market_price
                    free_float += sell_tokens
                    algo_executed = -sell_tokens
        algo_flow_tokens = algo_executed

        if crash_defense_enabled:
            crash_active_timer = max(0, int(algorithm_state.get("crash_active_timer", 0)))
            budget_remaining = float(algorithm_state.get("crash_budget_remaining", 0.0))
            if (crash_detection or crash_active_timer > 0) and budget_remaining > epsilon and treasury_cash > epsilon:
                crash_active_flag = True
                step_budget_cap = min(budget_remaining, treasury_cash)
                spend_fraction = np.clip(crash_cfg.aggression_pct / 100.0, 0.0, 1.0)
                per_step_spend = step_budget_cap * spend_fraction
                if crash_active_timer > 0:
                    per_step_spend = max(per_step_spend, budget_remaining / max(crash_active_timer, 1))
                per_step_spend = min(per_step_spend, step_budget_cap)
                gas_share = np.clip(crash_cfg.gas_subsidy_share_pct / 100.0, 0.0, 1.0)
                crash_gas_spend_step = per_step_spend * gas_share
                crash_buy_cash = per_step_spend - crash_gas_spend_step
                crash_buy_tokens_step = crash_buy_cash / max(market_price, epsilon)
                buy_cash_actual = 0.0
                if crash_buy_tokens_step > epsilon:
                    crash_buy_tokens_step = min(crash_buy_tokens_step, treasury_cash / max(market_price, epsilon))
                    buy_cash_actual = crash_buy_tokens_step * market_price
                    treasury_cash = max(0.0, treasury_cash - buy_cash_actual)
                    treasury_tokens += crash_buy_tokens_step
                    policy_flow += crash_buy_tokens_step
                else:
                    crash_buy_tokens_step = 0.0
                actual_gas_spend = 0.0
                if crash_gas_spend_step > 0:
                    actual_gas_spend = min(crash_gas_spend_step, treasury_cash)
                    treasury_cash = max(0.0, treasury_cash - actual_gas_spend)
                    gas_boost_tokens = (actual_gas_spend / max(market_price, epsilon)) * max(crash_cfg.gas_efficiency, 0.0)
                    if gas_boost_tokens > 0:
                        organic_flow += gas_boost_tokens
                crash_gas_spend_step = actual_gas_spend
                crash_cash_spend_step = max(0.0, buy_cash_actual + actual_gas_spend)
                budget_remaining = max(0.0, budget_remaining - crash_cash_spend_step)
                algorithm_state["crash_budget_remaining"] = budget_remaining
            else:
                crash_active_flag = crash_active_flag or crash_active_timer > 0
            if algorithm_state.get("crash_active_timer", 0) > 0:
                algorithm_state["crash_active_timer"] = max(0, int(algorithm_state["crash_active_timer"]) - 1)
            crash_budget_remaining = float(algorithm_state.get("crash_budget_remaining", 0.0))
        else:
            crash_detection = False
            crash_active_flag = False
            crash_cash_spend_step = 0.0
            crash_buy_tokens_step = 0.0
            crash_gas_spend_step = 0.0
            crash_budget_remaining = 0.0

        if central_bank_live and policy.module_gas_subsidy and gas_subsidy_pool > 0 and abs_dev <= hard_band:
            subsidy = min(gas_subsidy_pool, policy.gas_subsidy_rate * stage_multiplier * users)
            gas_subsidy_pool = max(0.0, gas_subsidy_pool - subsidy)
            organic_flow += (subsidy / max(nav_price, epsilon)) * 0.15

        organic_flow = float(np.clip(organic_flow, -max_flow, max_flow))

        transaction_volume_tokens = abs(organic_flow) + abs(attack_flow_tokens) + abs(policy_flow)
        tax_tokens = transaction_volume_tokens * tx_tax_rate
        burn_share = np.clip(supply_plan.burn_share_of_tax, 0.0, 1.0)
        vault_share = np.clip(supply_plan.burn_vault_share, 0.0, 1.0 - burn_share)
        burn_component = tax_tokens * burn_share
        vault_component = tax_tokens * vault_share
        burn_trigger_price = supply_plan.burn_pause_price * nav_price
        if market_price <= burn_trigger_price:
            burn_component = 0.0
        if burn_component > 0 and free_float > sim.supply_floor:
            burn_capacity = max(0.0, free_float - sim.supply_floor)
            actual_burn = min(burn_component, burn_capacity)
            free_float -= actual_burn
            burned_step += actual_burn
            burned_cumulative += actual_burn
        if vault_component > 0:
            burn_vault_tokens += vault_component

        escalate_price = supply_plan.burn_escalate_price * nav_price
        if market_price > escalate_price and free_float > sim.supply_floor:
            extra_burn = min(free_float - sim.supply_floor, free_float * 0.01 * max(1.0, supply_plan.goal_release_multiplier))
            if extra_burn > 0:
                free_float -= extra_burn
                burned_step += extra_burn
                burned_cumulative += extra_burn
        release_gate_price = min(
            supply_plan.burn_vault_release_threshold * max(supply_plan.goal_price, epsilon),
            max(burn_trigger_price, nav_price * 0.4),
        )
        if market_price < release_gate_price and burn_vault_tokens > 0:
            release_amount = burn_vault_tokens * np.clip(supply_plan.burn_vault_release_fraction, 0.0, 1.0)
            release_amount = min(release_amount, burn_vault_tokens)
            if release_amount > 0:
                burn_vault_tokens -= release_amount
                burn_vault_release_step = release_amount
                burn_vault_released += release_amount
                free_float += release_amount

        supply_flow_tokens = mint_to_market + unlock_tokens + burn_vault_release_step - burned_step
        policy_flow += supply_flow_tokens

        if supply_plan.absolute_supply_enabled:
            target_supply = supply_plan.absolute_supply_tokens
            if target_supply is None:
                target_supply = sim.initial_free_float + sim.founder_locked + sim.initial_treasury_tokens
            target_supply = max(float(target_supply), sim.supply_floor)
            current_supply = free_float + locked_supply + treasury_tokens
            delta_supply = current_supply - target_supply
            if abs(delta_supply) > 1e-6:
                if delta_supply > 0:
                    burn_float = min(delta_supply, max(0.0, free_float - sim.supply_floor))
                    if burn_float > 0:
                        free_float -= burn_float
                        burned_step += burn_float
                        burned_cumulative += burn_float
                        absolute_supply_adjustment_step -= burn_float
                        delta_supply -= burn_float
                    if delta_supply > 0 and treasury_tokens > 0:
                        burn_treasury = min(delta_supply, treasury_tokens)
                        treasury_tokens -= burn_treasury
                        absolute_supply_adjustment_step -= burn_treasury
                        delta_supply -= burn_treasury
                    if delta_supply > 0 and locked_supply > 0:
                        reduce_locked = min(delta_supply, locked_supply)
                        locked_supply -= reduce_locked
                        absolute_supply_adjustment_step -= reduce_locked
                        delta_supply -= reduce_locked
                else:
                    mint_amount = -delta_supply
                    treasury_tokens += mint_amount
                    minted_step += mint_amount
                    minted_cumulative += mint_amount
                    absolute_supply_adjustment_step += mint_amount
                total_supply = free_float + locked_supply + treasury_tokens
                policy_flow += absolute_supply_adjustment_step

        treasury_nav = treasury_tokens * market_price + treasury_cash
        bankruptcy_flow = 0.0
        bankruptcy_triggered = False
        if central_bank_live and treasury_nav <= 1.0:
            policy_bankrupt = True
            if policy_bankrupt_step is None:
                policy_bankrupt_step = step
                bankruptcy_triggered = True

        if policy_bankrupt and bankruptcy_triggered:
            liquidity_depth = max(liquidity_depth * (1 - policy.bankruptcy_liquidity_hit), 10_000.0)
            effective_depth = max(effective_depth * (1 - policy.bankruptcy_liquidity_hit), 10_000.0)
            panic_tokens = policy.bankruptcy_selloff * max(liquidity_depth / max(market_price, epsilon), 0.0)
            bankruptcy_flow = -panic_tokens
            organic_flow += bankruptcy_flow
            confidence = max(sim.confidence_floor, confidence - policy.bankruptcy_confidence_hit)
            stage_multiplier = 0.0
            policy_stage_active = False
            central_bank_live = False

        savings_pool = max(0.0, savings_pool + savings_tokens * nav_price)

        net_flow_tokens = organic_flow + attack_flow_tokens + policy_flow
        effective_net_flow = 0.65 * net_flow_tokens + 0.35 * net_flow_memory
        net_flow_memory = effective_net_flow
        depth_tokens = max(effective_depth / max(market_price, epsilon), 1.0)
        flow_pressure = np.clip(effective_net_flow / depth_tokens, -4.0, 4.0)
        flow_return = sim.impact_coeff * 0.65 * np.tanh(flow_pressure)

        anchor_components: List[Tuple[float, float]] = []
        anchor_mode = algo_modules.peg_anchor if (algorithm_cfg.enabled or algorithm_cfg.master_switch) else "hybrid"
        if anchor_mode == "gold":
            anchor_components.append((nav_price, 1.0))
            anchor_components.append((fundamental_price, 0.4))
        elif anchor_mode == "fundamental":
            anchor_components.append((fundamental_price, 1.3))
            anchor_components.append((nav_price, 0.2))
        elif anchor_mode == "none":
            anchor_components.append((fundamental_price, 1.0))
        else:
            anchor_components.append((fundamental_price, 1.0))
            anchor_components.append((nav_price, 0.5))
        gold_guidance = max(sim.gold_guidance_strength, 0.0)
        if gold_guidance > 0:
            anchor_components.append((nav_price, gold_guidance))
        readiness_nav_weight = 0.6 * peg_readiness
        if central_bank_live and readiness_nav_weight > 0:
            anchor_components.append((nav_price, readiness_nav_weight))
        fundamental_mispricing = abs(math.log(max(market_price, epsilon) / max(fundamental_price, epsilon)))
        reversion_strength = np.clip(
            sim.baseline_reversion + peg_readiness * 0.15 + max(0.0, fundamental_mispricing - 0.35) * 0.3,
            0.0,
            0.9,
        )
        if algorithm_active and algo_modules.alpha_stabilizer:
            reversion_strength = float(np.clip(reversion_strength + abs(algorithm_signal) * 0.08, 0.0, 0.95))
        if central_bank_live:
            policy_weight = (2.5 + policy.reversion_bonus * 1.5) * max(0.3, stage_multiplier)
            anchor_components.append((nav_price, policy_weight))
            bonus_scale = max(policy.nav_band_soft, 1e-4)
            scaled_bonus = policy.reversion_bonus * stage_multiplier * min(1.5, abs_dev / bonus_scale)
            reversion_strength = min(0.95, reversion_strength + scaled_bonus)
        anchor_price = sum(price * weight for price, weight in anchor_components) / sum(weight for _, weight in anchor_components)
        mispricing = math.log(max(market_price, epsilon) / max(anchor_price, epsilon))
        reversion_return = -reversion_strength * np.tanh(mispricing * 1.1)

        nav_return = math.log(nav_price / max(prev_nav_price, epsilon))
        gold_coupling = 0.12 + (0.25 * stage_multiplier if central_bank_live else 0.06)
        if anchor_mode == "none":
            gold_coupling *= 0.2
        elif anchor_mode == "fundamental":
            gold_coupling *= 0.4
        elif anchor_mode == "gold":
            gold_coupling *= 1.3
        gold_pull = gold_coupling * nav_return

        policy_pull = 0.0
        if central_bank_live:
            nav_gap = math.log(nav_price / max(market_price, epsilon))
            policy_pull = np.tanh(nav_gap * 1.2) * (0.18 + 0.32 * stage_multiplier)
            if abs_dev <= soft_band:
                policy_pull *= 1.6
            else:
                policy_pull *= min(1.4, abs_dev / max(soft_band, 1e-4))
            policy_pull *= min(0.55, abs(nav_gap))

        momentum_return = 0.035 * np.tanh(price_momentum * 1.6)
        noise_scale = 0.01 + sim.organic_noise * 0.5 + 0.007 * hype_index
        noise_return = rng.gauss(0.0, noise_scale)

        total_log_return = flow_return + reversion_return + gold_pull + policy_pull + noise_return + momentum_return
        mispricing_ratio = abs(math.log(max(market_price, epsilon) / max(anchor_price, epsilon)))
        down_clip = -0.25 - min(0.85, mispricing_ratio * 0.5 + fundamental_mispricing * 0.35)
        up_clip = 0.2 + min(0.15, max(0.0, price_momentum) * 0.1)
        down_clip = max(-1.5, down_clip)
        up_clip = min(0.35, up_clip)
        total_log_return = float(np.clip(total_log_return, down_clip, up_clip))
        updated_price = max(epsilon, market_price * math.exp(total_log_return))
        min_price = max(fundamental_price * 0.4, intrinsic_floor * 0.3, epsilon)
        max_price = nav_price * 2.5
        blend_weight = 0.6 - min(0.5, mispricing_ratio * 0.2 + fundamental_mispricing * 0.1)
        blend_weight = float(np.clip(blend_weight, 0.1, 0.6))
        market_price = blend_weight * prev_price + (1 - blend_weight) * updated_price
        market_price = float(np.clip(market_price, min_price, max_price))
        deviation = (market_price - nav_price) / max(nav_price, epsilon)
        abs_dev = abs(deviation)
        fundamental_gap = (market_price - fundamental_price) / max(fundamental_price, epsilon)

        user_tokens = np.clip(user_tokens + organic_flow, 0.0, free_float)
        confidence_delta = -np.tanh(fundamental_gap * sim.confidence_sensitivity) * 0.03
        confidence_delta += 0.015 * np.tanh((adoption_rate - churn_rate) * 8)
        if macro_event:
            confidence_delta -= macro_penalty * 0.25
        if policy_bankrupt:
            confidence_delta -= 0.08
        confidence_delta += rng.gauss(0, 0.01)
        confidence = float(np.clip(confidence + confidence_delta, sim.confidence_floor, sim.confidence_ceiling))

        total_supply = free_float + locked_supply + treasury_tokens
        treasury_circulating = treasury_tokens * (0.25 if central_bank_live else 0.05)
        circulating_supply = max(sim.supply_floor, free_float + treasury_circulating)
        circulating_supply = min(circulating_supply, total_supply)
        market_cap = market_price * circulating_supply
        fully_diluted_cap = market_price * total_supply
        if growth_auto_enabled:
            algorithm_state["oracle_prev_cap"] = float(market_cap)
            algorithm_state["oracle_prev_price"] = float(market_price)

        attacker_cash_total = sum(state.cash for state in attacker_states)
        attacker_tokens_total = sum(state.tokens for state in attacker_states)
        attacker_position_value_total = attacker_tokens_total * market_price
        attacker_capital_total = max(attacker_base_capital, 1.0)
        attacker_pnl = attacker_cash_total + attacker_position_value_total - attacker_capital_total
        avg_progress = (
            float(np.mean([state.objective_progress for state in attacker_states])) if attacker_states else 0.0
        )
        avg_intensity = float(np.mean(np.abs(intent_samples))) if intent_samples else 0.0

        central_bank_active = central_bank_live

        if growth_auto_enabled and growth_oracle_enabled:
            history = list(algorithm_state.get("growth_oracle_history", []))
            goal_metric = market_cap if growth_oracle_goal == "market_cap" else market_price
            history.append(float(goal_metric))
            hist_cap = max(growth_oracle_horizon, 2)
            if len(history) > hist_cap:
                history = history[-hist_cap:]
            if len(history) >= 2:
                idx = np.arange(len(history), dtype=float)
                try:
                    slope = float(np.polyfit(idx, history, 1)[0])
                except np.linalg.LinAlgError:
                    slope = 0.0
                scale = max(abs(history[-1]), 1.0)
                normalized = slope * growth_oracle_horizon / scale
                oracle_bias = float(np.clip(normalized, -2.5, 2.5))
            else:
                oracle_bias = float(algorithm_state.get("growth_oracle_bias", 0.0) * 0.8)
            algorithm_state["growth_oracle_history"] = history
            algorithm_state["growth_oracle_bias"] = oracle_bias
        elif growth_oracle_enabled:
            algorithm_state["growth_oracle_bias"] = float(algorithm_state.get("growth_oracle_bias", 0.0) * 0.8)

        if growth_auto_enabled:
            prev_cash_level = algorithm_state.get("growth_prev_cash", treasury_cash)
            growth_cash_delta = treasury_cash - prev_cash_level
            tolerance_cash = growth_cash_tolerance * max(prev_cash_level, 1.0)
            dynamic_rate_next = float(np.clip(algorithm_state.get("growth_dynamic_tax_rate", tx_tax_rate), growth_min_tax, growth_max_tax))
            dynamic_rate_next = float(np.clip(dynamic_rate_next + oracle_fee_adjust, growth_min_tax, growth_max_tax))
            buy_bias_next = oracle_bias_delta
            if growth_cash_delta < -tolerance_cash:
                dynamic_rate_next = min(growth_max_tax, dynamic_rate_next + growth_fee_step)
                bleed_ratio = min(1.0, abs(growth_cash_delta) / max(prev_cash_level, 1.0))
                buy_bias_next = min(buy_bias_next, 0.0) + (-growth_buy_brake * bleed_ratio)
            elif growth_cash_delta > tolerance_cash:
                dynamic_rate_next = max(growth_min_tax, dynamic_rate_next - growth_fee_step)
                flush_ratio = min(1.0, growth_cash_delta / max(prev_cash_level, 1.0))
                buy_bias_next = max(buy_bias_next, 0.0) + (growth_buy_push * flush_ratio)
            else:
                base_rate = supply_plan.tx_tax_rate
                dynamic_rate_next = dynamic_rate_next * 0.9 + base_rate * 0.1
            buy_bias_next = float(np.clip(buy_bias_next, -growth_buy_brake, growth_buy_push))
            algorithm_state["growth_prev_cash"] = treasury_cash
            algorithm_state["growth_dynamic_tax_rate"] = float(np.clip(dynamic_rate_next, growth_min_tax, growth_max_tax))
            algorithm_state["growth_buy_bias"] = buy_bias_next
            algorithm_state["growth_cash_delta"] = growth_cash_delta
            growth_oracle_bias = float(np.clip(buy_bias_next / max(growth_buy_push, 1e-6), -2.5, 2.5))
            algorithm_state["growth_oracle_bias"] = growth_oracle_bias
            algorithm_state["oracle_mint_bias"] = oracle_mint_scale
            if oracle_hype_boost and lever_controls.get("hype_boost", True):
                algorithm_state["oracle_hype_boost"] = oracle_hype_boost
        else:
            growth_cash_delta = 0.0
            algorithm_state.pop("growth_cash_delta", None)
            algorithm_state.pop("growth_prev_cash", None)
            algorithm_state.pop("growth_dynamic_tax_rate", None)
            algorithm_state.pop("growth_buy_bias", None)
            algorithm_state.pop("growth_oracle_bias", None)
            algorithm_state.pop("oracle_mint_bias", None)
            algorithm_state.pop("oracle_hype_boost", None)

        growth_oracle_bias = float(algorithm_state.get("growth_oracle_bias", 0.0)) if growth_auto_enabled else 0.0

        df_rows.append(
            {
                "step": step,
                "month": month,
                "year": time_years,
                "gold_price": nav_price,
                "token_price": market_price,
                "nav_price": nav_price,
                "peg_deviation_pct": (market_price / max(nav_price, epsilon)) - 1,
                "users": users,
                "liquidity_depth": liquidity_depth,
                "hold_share": hold_share,
                "confidence": confidence,
                "organic_flow_tokens": organic_flow,
                "bootstrap_flow_tokens": bootstrap_flow,
                "retail_capital": retail_capital,
                "whale_capital": whale_capital,
                "speculator_capital": speculative_capital,
                "community_capital": community_capital,
                "capital_price": capital_price,
                "policy_flow_tokens": policy_flow,
                "attack_flow_tokens": attack_flow_tokens,
                "net_flow_tokens": net_flow_tokens,
                "free_float_supply": free_float,
                "locked_supply": locked_supply,
                "treasury_tokens": treasury_tokens,
                "treasury_cash": treasury_cash,
                "algo_trade_tokens": algo_executed,
                "growth_dynamic_tax_rate": dynamic_tax_rate,
                "growth_cash_delta": growth_cash_delta,
                "growth_buy_bias": growth_buy_bias,
                "growth_oracle_bias": growth_oracle_bias if growth_auto_enabled and growth_oracle_enabled else 0.0,
                "growth_baseline_cash": float(algorithm_state.get("growth_baseline_cash", treasury_cash)),
                "growth_surplus_cash_step": growth_surplus_cash_step,
                "growth_surplus_spend_tokens_step": growth_surplus_spend_tokens_step,
                "growth_surplus_spend_cash_step": growth_surplus_spend_cash_step,
                "growth_surplus_bucket": growth_surplus_bucket,
                "growth_fee_cash_step": growth_fee_cash_step,
                "absolute_supply_adjustment_step": absolute_supply_adjustment_step,
                "gas_subsidy_pool": gas_subsidy_pool,
                "savings_pool": savings_pool,
                "crash_defense_active": 1 if crash_active_flag else 0,
                "crash_defense_detected": 1 if crash_detection else 0,
                "crash_defense_cash_spent": crash_cash_spend_step,
                "crash_defense_buy_tokens": crash_buy_tokens_step,
                "crash_defense_gas_spend": crash_gas_spend_step,
                "crash_defense_budget_remaining": crash_budget_remaining,
                "crash_defense_drop_pct": crash_drop_pct,
                "circulating_supply": circulating_supply,
                "minted_step": minted_step,
                "burned_step": burned_step,
                "unlock_step": unlock_tokens,
                "minted_cumulative": minted_cumulative,
                "burned_cumulative": burned_cumulative,
                "burn_vault_release_step": burn_vault_release_step,
                "burn_vault_balance": burn_vault_tokens,
                "burn_vault_cumulative": burn_vault_released,
                "active_attacks": ", ".join(a.name for a in active_attacks),
                "adoption_rate": adoption_rate,
                "churn_rate": churn_rate,
                "new_users": new_users,
                "lost_users": lost_users,
                "sticky_users": sticky_users,
                "hype_new_users": event_new_users,
                "effective_liquidity": effective_depth,
                "savings_tokens": savings_tokens,
                "gas_subsidy": subsidy,
                "fundamental_price": fundamental_price,
                "anchor_price": anchor_price,
                "fundamental_gap": fundamental_gap,
                "total_supply": total_supply,
                "market_cap": market_cap,
                "fully_diluted_market_cap": fully_diluted_cap,
                "treasury_nav": treasury_nav,
                "central_bank_active": central_bank_active,
                "policy_bankrupt": policy_bankrupt,
                "policy_stage_multiplier": stage_multiplier,
                "policy_stage_active": policy_stage_active,
                "macro_event": macro_event,
                "macro_penalty": macro_penalty,
                "financing_cash": financing_cash,
                "bankruptcy_flow_tokens": bankruptcy_flow,
                "hype_index": hype_index,
                "event_buy_tokens": event_buy_tokens,
                "event_sell_tokens": event_sell_tokens,
                "transaction_tax_tokens": tax_tokens,
                "tx_tax_rate": tx_tax_rate,
                "growth_surplus_spend_tokens": growth_surplus_spend_tokens_step,
                "auto_flow_tokens": auto_flow_tokens,
                "supply_flow_tokens": supply_flow_tokens,
                "attacker_auto_signal": auto_signal,
                "attacker_auto_mode": auto_mode,
                "attacker_objective_progress": avg_progress,
                "attacker_intensity": avg_intensity,
                "algorithm_signal": algorithm_signal,
                "algorithm_goal_progress": algorithm_goal_progress,
                "algorithm_flow_tokens": algo_flow_tokens,
                "attacker_cash_total": attacker_cash_total,
                "attacker_tokens_total": attacker_tokens_total,
                "attacker_capital_total": attacker_capital_total,
                "attacker_position_value_total": attacker_position_value_total,
            }
        )

        attacker_state_rows.append(
            {
                "step": step,
                "month": month,
                "auto_signal": auto_signal,
                "auto_mode": auto_mode,
                "auto_flow_tokens": auto_flow_tokens,
                "attacker_cash": attacker_cash_total,
                "attacker_tokens": attacker_tokens_total,
                "attacker_position_value": attacker_position_value_total,
                "attacker_pnl": attacker_pnl,
                "attacker_intensity": avg_intensity,
                "attacker_objective_progress": avg_progress,
                "attacker_price": market_price,
            }
        )

    timeline = pd.DataFrame(df_rows)
    attacker_summary = pd.DataFrame(trade_rows)
    if not attacker_summary.empty:
        attacker_summary["year"] = attacker_summary["step"] / sim.steps_per_year
        attacker_summary["month"] = attacker_summary["step"] + 1
        attacker_summary["tokens_abs"] = attacker_summary["tokens"].abs()
    attacker_state_df = pd.DataFrame(attacker_state_rows)

    return SimulationOutput(
        timeline=timeline,
        attacker_trades=attacker_summary,
        attacker_state=attacker_state_df,
        config=sim,
        policy=policy,
        algorithm=algorithm_cfg,
        attacks=attack_events,
        attacker_settings=attacker_settings,
    )


def render_simulation_tab(output: SimulationOutput) -> None:
    df = output.timeline
    state_df = output.attacker_state if output.attacker_state is not None else pd.DataFrame()
    if "hype_index" not in df.columns:
        df["hype_index"] = 0.0
    df["hold_share_pct"] = df.get("hold_share", 0.0) * 100
    df["minted_step"] = df.get("minted_step", 0.0)
    df["algo_trade_tokens"] = df.get("algo_trade_tokens", 0.0)
    df["algorithm_flow_tokens"] = df.get("algorithm_flow_tokens", 0.0)
    df["algorithm_signal"] = df.get("algorithm_signal", 0.0)
    df["algorithm_goal_progress"] = df.get("algorithm_goal_progress", 0.0)
    df["burned_step"] = df.get("burned_step", 0.0)
    df["burned_step_negative"] = -df["burned_step"]
    df["unlock_step"] = df.get("unlock_step", 0.0)
    df["burn_vault_release_step"] = df.get("burn_vault_release_step", 0.0)
    df["burn_vault_balance"] = df.get("burn_vault_balance", 0.0)
    df["transaction_tax_tokens"] = df.get("transaction_tax_tokens", 0.0)
    df["event_buy_tokens"] = df.get("event_buy_tokens", 0.0)
    df["event_sell_tokens"] = df.get("event_sell_tokens", 0.0)
    df["tx_tax_rate"] = df.get("tx_tax_rate", 0.0)
    df["growth_cash_delta"] = df.get("growth_cash_delta", 0.0)
    df["growth_surplus_spend_cash_step"] = df.get("growth_surplus_spend_cash_step", 0.0)
    df["growth_fee_cash_step"] = df.get("growth_fee_cash_step", 0.0)
    df["growth_oracle_bias"] = df.get("growth_oracle_bias", 0.0)
    df["absolute_supply_adjustment_step"] = df.get("absolute_supply_adjustment_step", 0.0)
    df["attacker_cash_total"] = df.get("attacker_cash_total", 0.0)
    df["attacker_tokens_total"] = df.get("attacker_tokens_total", 0.0)
    df["attacker_position_value_total"] = df.get("attacker_position_value_total", 0.0)
    df["sticky_users"] = df.get("sticky_users", df.get("users", 0.0) * 0.3)
    df["crash_defense_active"] = df.get("crash_defense_active", 0)
    df["crash_defense_detected"] = df.get("crash_defense_detected", 0)
    df["crash_defense_cash_spent"] = df.get("crash_defense_cash_spent", 0.0)
    df["crash_defense_buy_tokens"] = df.get("crash_defense_buy_tokens", 0.0)
    df["crash_defense_gas_spend"] = df.get("crash_defense_gas_spend", 0.0)
    df["crash_defense_budget_remaining"] = df.get("crash_defense_budget_remaining", 0.0)
    df["crash_defense_drop_pct"] = df.get("crash_defense_drop_pct", 0.0)
    autopilot_cfg = getattr(output.algorithm, "custom_params", {}).get("growth_autopilot", {}) if output.algorithm else {}
    autopilot_enabled = bool(autopilot_cfg.get("enabled", False))
    final_price = float(df["token_price"].iloc[-1])
    final_peg_dev = float(df["peg_deviation_pct"].iloc[-1]) * 100
    final_treasury_cash = float(df["treasury_cash"].iloc[-1])
    final_treasury_tokens = float(df["treasury_tokens"].iloc[-1])
    summary_cols = st.columns(4)
    summary_cols[0].metric("Final token price", f"${final_price:,.2f}")
    summary_cols[1].metric("Final peg deviation", f"{final_peg_dev:+.2f}%")
    summary_cols[2].metric("Treasury cash", f"${final_treasury_cash:,.0f}")
    summary_cols[3].metric("Treasury tokens", f"{final_treasury_tokens:,.0f}")

    if autopilot_enabled:
        fee_capture_pct = float(autopilot_cfg.get("fee_capture_rate", 0.0)) * 100
        surplus_spend_pct = float(autopilot_cfg.get("surplus_spend_rate_pct", 0.0))
        cash_buffer_pct = float(autopilot_cfg.get("cash_buffer_pct", 0.0))
        growth_oracle_enabled = bool(autopilot_cfg.get("oracle_enabled", False))
        growth_oracle_horizon = int(autopilot_cfg.get("oracle_horizon_steps", 0))
        growth_oracle_goal = str(autopilot_cfg.get("target_metric", autopilot_cfg.get("oracle_goal", "market_cap")))
        growth_oracle_weight = float(autopilot_cfg.get("oracle_weight", 1.0))
        growth_oracle_accuracy = float(autopilot_cfg.get("oracle_accuracy", 0.0))
        focus_label_map = {
            "market_cap": "focus: maximise market cap",
            "token_price": "focus: maximise token price",
            "hybrid": "focus: hybrid (cap + price)",
        }
        focus_text = focus_label_map.get(growth_oracle_goal, f"focus: {growth_oracle_goal}")
        fee_cash = float(df["growth_fee_cash_step"].sum())
        spend_cash = float(df["growth_surplus_spend_cash_step"].sum())
        buy_tokens = float(df["algo_trade_tokens"].clip(lower=0).sum())
        sell_tokens = float(df["algo_trade_tokens"].clip(upper=0).abs().sum())
        trade_steps = int((df["algo_trade_tokens"].abs() > 1e-6).sum())
        st.markdown(
            f"**Growth autopilot** — fee capture {fee_capture_pct:.2f}% | surplus spend {surplus_spend_pct:.1f}% per step | "
            f"cash buffer {cash_buffer_pct:.1f}%"
        )
        autopilot_cols = st.columns(3)
        autopilot_cols[0].metric("Cash skimmed", f"${fee_cash:,.0f}")
        autopilot_cols[1].metric("Cash redeployed", f"${spend_cash:,.0f}")
        autopilot_cols[2].metric("Net tokens acquired", f"{buy_tokens - sell_tokens:,.0f}", delta=f"{trade_steps} steps")
        if growth_oracle_enabled:
            st.caption(
                f"{focus_text} • oracle accuracy {growth_oracle_accuracy:.2f}"
            )
            st.caption(
                f"Oracle assist: {growth_oracle_horizon} step lookahead focused on {growth_oracle_goal.replace('_', ' ')} "
                f"(bias {growth_oracle_weight:.2f})"
            )
        else:
            st.caption(focus_text)
    else:
        st.markdown("**Growth autopilot** is off for this run. Enable it in Algorithm Forge if you want automated fee skim and buybacks.")

    st.divider()

    st.markdown("### Price and peg")
    st.caption(
        "`Token price` shows the market print, `Fundamental price` is the model's estimate of fair value, and `Gold price` is the external anchor."
    )
    fig = px.line(
        df,
        x="month",
        y=["token_price", "fundamental_price", "gold_price"],
        labels={"value": "Price (USD)", "variable": "Series"},
        template="plotly_white",
    )
    # Set gold_price to be disabled by default
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name == "gold_price" else None)
    _plotly(fig, full_width=True)

    st.markdown("#### Peg deviation")
    st.caption("Values above zero mean the token trades rich to gold; below zero means it trades at a discount.")
    dev_fig = px.line(
        df,
        x="month",
        y=df["peg_deviation_pct"] * 100,
        labels={"y": "Deviation (%)", "month": "Month"},
        template="plotly_white",
    )
    dev_fig.add_hline(y=2, line_dash="dash", line_color="red")
    dev_fig.add_hline(y=-2, line_dash="dash", line_color="red")
    _plotly(dev_fig, full_width=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Users and liquidity")
        st.caption("Track adoption alongside the depth of the order book. A healthy system grows both.")
        # Use dual y-axis: users on left, liquidity on right
        ul_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        ul_fig.add_trace(
            go.Scatter(x=df["month"], y=df["users"], name="Users", mode="lines"),
            secondary_y=False,
        )
        ul_fig.add_trace(
            go.Scatter(x=df["month"], y=df["liquidity_depth"], name="Liquidity depth", mode="lines"),
            secondary_y=True,
        )
        
        ul_fig.update_xaxes(title_text="Month")
        ul_fig.update_yaxes(title_text="Users", secondary_y=False)
        ul_fig.update_yaxes(title_text="Liquidity (USD)", secondary_y=True)
        ul_fig.update_layout(template="plotly_white", hovermode="x unified")
        
        _plotly(ul_fig, full_width=True)

    with col2:
        st.markdown("#### Treasury inventories")
        st.caption("Tokens and cash owned by the treasury. Draining either too far means your algorithm is running out of ammunition.")
        treasury_fig = make_subplots(specs=[[{"secondary_y": True}]])
        treasury_fig.add_trace(
            go.Scatter(x=df["month"], y=df["treasury_cash"], name="Treasury cash (USD)", mode="lines"),
            secondary_y=False,
        )
        treasury_fig.add_trace(
            go.Scatter(x=df["month"], y=df["treasury_tokens"], name="Treasury tokens", mode="lines"),
            secondary_y=True,
        )
        treasury_fig.update_xaxes(title_text="Month")
        treasury_fig.update_yaxes(title_text="Cash (USD)", secondary_y=False, separatethousands=True, tickprefix="$")
        treasury_fig.update_yaxes(title_text="Tokens", secondary_y=True, separatethousands=True)
        treasury_fig.update_layout(template="plotly_white", hovermode="x unified")
        _plotly(treasury_fig, full_width=True)
    st.divider()

    st.markdown("### Sentiment and hype")
    st.caption("Hold share measures how much of the float is parked by long-term believers; the hype index captures the narrative heat.")
    # Use dual y-axis for better scaling
    hold_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    hold_fig.add_trace(
        go.Scatter(x=df["month"], y=df["hold_share_pct"], name="Hold share (%)", mode="lines"),
        secondary_y=False,
    )
    hold_fig.add_trace(
        go.Scatter(x=df["month"], y=df["hype_index"], name="Hype index", mode="lines"),
        secondary_y=True,
    )
    
    hold_fig.update_xaxes(title_text="Month")
    hold_fig.update_yaxes(title_text="Hold Share (%)", secondary_y=False)
    hold_fig.update_yaxes(title_text="Hype Index", secondary_y=True)
    hold_fig.update_layout(template="plotly_white", hovermode="x unified")
    
    _plotly(hold_fig, full_width=True)
    st.divider()

    st.markdown("### User growth dynamics")
    st.caption("Adoption adds wallets, churn removes them. Watch the balance to see whether your story keeps the community growing.")
    growth_df = df.copy()
    growth_df["adoption_rate_pct"] = growth_df["adoption_rate"] * 100
    growth_df["churn_rate_pct"] = growth_df["churn_rate"] * 100
    growth_fig = px.line(
        growth_df,
        x="month",
        y=["adoption_rate_pct", "churn_rate_pct"],
        labels={"value": "Rate (%)", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(growth_fig, full_width=True)

    growth_df["net_users"] = df["new_users"] - df["lost_users"]
    net_fig = px.bar(
        growth_df,
        x="month",
        y="net_users",
        labels={"y": "Net new users", "month": "Month"},
        template="plotly_white",
    )
    _plotly(net_fig, full_width=True)
    st.divider()

    st.markdown("### Balance sheet")
    cap_fig = px.line(
        df,
        x="month",
        y=["market_cap", "fully_diluted_market_cap", "treasury_nav"],
        labels={"value": "USD", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(cap_fig, full_width=True)

    supply_fig = px.line(
        df,
        x="month",
        y=["free_float_supply", "circulating_supply", "total_supply"],
        labels={"value": "Tokens", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(supply_fig, full_width=True)

    st.markdown("### Capital pools")
    capital_fig = px.line(
        df,
        x="month",
        y=["retail_capital", "whale_capital", "speculator_capital", "community_capital"],
        labels={"value": "Capital (USD)", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(capital_fig, full_width=True)
    st.divider()

    st.markdown("### Supply flows")
    supply_plot_df = df[["month"]].copy()
    supply_plot_df["Minted"] = df["minted_step"]
    supply_plot_df["Unlocked"] = df["unlock_step"]
    supply_plot_df["Burned"] = -df["burned_step"]
    supply_plot_df["Vault release"] = df["burn_vault_release_step"]
    supply_plot_df["Absolute adjust"] = df.get("absolute_supply_adjustment_step", 0.0)
    supply_long = supply_plot_df.melt(id_vars="month", var_name="Flow", value_name="tokens")
    supply_fig = px.bar(
        supply_long,
        x="month",
        y="tokens",
        color="Flow",
        labels={"tokens": "Tokens", "Flow": "Series"},
        template="plotly_white",
    )
    _plotly(supply_fig, full_width=True)

    vault_fig = px.line(
        df,
        x="month",
        y="burn_vault_balance",
        labels={"burn_vault_balance": "Vault balance (tokens)", "month": "Month"},
        template="plotly_white",
        title="Burn vault balance",
    )
    _plotly(vault_fig, full_width=True)
    st.divider()

    st.markdown("### External pressure and flows")
    hype_attack_fig = px.line(
        df,
        x="month",
        y=["event_buy_tokens", "event_sell_tokens", "attack_flow_tokens", "auto_flow_tokens", "transaction_tax_tokens"],
        labels={"value": "Tokens", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(hype_attack_fig, full_width=True)

    st.markdown("#### Attacker footprint")
    att_cash_fig = px.line(
        df,
        x="month",
        y=["attacker_cash_total", "attacker_position_value_total"],
        labels={"value": "USD", "variable": "Series", "month": "Month"},
        template="plotly_white",
    )
    _plotly(att_cash_fig, full_width=True)

    inventory_fig = px.line(
        df,
        x="month",
        y="attacker_tokens_total",
        labels={"attacker_tokens_total": "Attacker inventory (tokens)", "month": "Month"},
        template="plotly_white",
    )
    _plotly(inventory_fig, full_width=True)
    st.divider()

    st.markdown("### Market vs fundamental gap")
    gap_fig = px.line(
        df,
        x="month",
        y=df["fundamental_gap"] * 100,
        labels={"y": "Deviation from fundamental (%)", "month": "Month"},
        template="plotly_white",
    )
    gap_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    _plotly(gap_fig, full_width=True)
    st.divider()

    st.markdown("### Central bank regime")
    regime_df = df.copy()
    regime_df["central_bank_flag"] = regime_df["central_bank_active"].astype(int)
    regime_fig = px.line(
        regime_df,
        x="month",
        y=["policy_stage_multiplier", "central_bank_flag"],
        labels={"value": "Stage / Active", "variable": "Series"},
        template="plotly_white",
    )
    _plotly(regime_fig, full_width=True)
    if regime_df["central_bank_flag"].sum() == 0:
        st.caption("Central bank modules were never activated during this run.")

    financing_fig = px.bar(
        regime_df,
        x="month",
        y="financing_cash",
        labels={"y": "Financing inflow (USD)", "month": "Month"},
        template="plotly_white",
    )
    _plotly(financing_fig, full_width=True)

    st.markdown("### Net token flows")
    flow_fig = px.line(
        df,
        x="month",
        y=["organic_flow_tokens", "policy_flow_tokens", "attack_flow_tokens", "algorithm_flow_tokens", "net_flow_tokens"],
        labels={"value": "Tokens", "variable": "Flow type"},
        template="plotly_white",
    )
    _plotly(flow_fig, full_width=True)
    st.divider()

    st.markdown("### Algorithm signal")
    algo_metric_fig = make_subplots(specs=[[{"secondary_y": True}]])
    algo_metric_fig.add_trace(
        go.Scatter(x=df["month"], y=df["algorithm_signal"], name="Algorithm signal", mode="lines"),
        secondary_y=False,
    )
    algo_metric_fig.add_trace(
        go.Scatter(x=df["month"], y=df["algorithm_goal_progress"], name="Goal progress", mode="lines"),
        secondary_y=True,
    )
    algo_metric_fig.update_xaxes(title_text="Month")
    algo_metric_fig.update_yaxes(title_text="Signal", secondary_y=False)
    algo_metric_fig.update_yaxes(title_text="Progress", secondary_y=True, range=[0, 1.5])
    algo_metric_fig.update_layout(template="plotly_white", hovermode="x unified")
    _plotly(algo_metric_fig, full_width=True)

    algo_flow_fig = px.bar(
        df,
        x="month",
        y="algorithm_flow_tokens",
        labels={"algorithm_flow_tokens": "Algorithm flow (tokens)", "month": "Month"},
        template="plotly_white",
    )
    _plotly(algo_flow_fig, full_width=True)

    move_colors = np.where(df["algo_trade_tokens"] >= 0, "#2ecc71", "#e74c3c")
    move_hover = np.column_stack((df["algo_trade_tokens"], df["algorithm_signal"], df["algorithm_goal_progress"]))
    algo_moves_fig = go.Figure()
    algo_moves_fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["algo_trade_tokens"],
            mode="markers",
            marker=dict(color=move_colors, size=9, line=dict(width=0)),
            name="Net tokens traded",
            customdata=move_hover,
            hovertemplate="Month %{x}<br>Trade %{customdata[0]:,.2f} tokens\n<br>Signal %{customdata[1]:+.2f}\n<br>Goal progress %{customdata[2]:.2f}<extra></extra>",
        )
    )
    algo_moves_fig.add_hline(y=0, line_dash="dot", line_color="#7f8c8d")
    algo_moves_fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Net algorithm tokens",
        title_text="Algorithm moves timeline",
    )
    _plotly(algo_moves_fig, full_width=True)

    crash_mask = (df["crash_defense_active"] > 0) | (df["crash_defense_detected"] > 0)
    if crash_mask.any():
        gas_token_equiv = np.where(df["token_price"] > 0, df["crash_defense_gas_spend"] / df["token_price"].replace(0, np.nan), 0.0)
        gas_token_equiv = np.nan_to_num(gas_token_equiv, nan=0.0, posinf=0.0, neginf=0.0)
        budget_token_equiv = np.where(df["token_price"] > 0, df["crash_defense_budget_remaining"] / df["token_price"].replace(0, np.nan), 0.0)
        budget_token_equiv = np.nan_to_num(budget_token_equiv, nan=0.0, posinf=0.0, neginf=0.0)
        crash_fig = make_subplots(specs=[[{"secondary_y": True}]])
        crash_fig.add_trace(
            go.Bar(
                x=df["month"],
                y=df["crash_defense_buy_tokens"],
                name="Crash buys (tokens)",
                marker_color="#1abc9c",
            ),
            secondary_y=False,
        )
        crash_fig.add_trace(
            go.Bar(
                x=df["month"],
                y=gas_token_equiv,
                name="Gas boost (token equiv)",
                marker_color="#3498db",
            ),
            secondary_y=False,
        )
        crash_fig.add_trace(
            go.Scatter(
                x=df["month"],
                y=budget_token_equiv,
                name="Budget remaining (token equiv)",
                mode="lines",
                line=dict(color="#95a5a6", dash="dot"),
            ),
            secondary_y=False,
        )
        crash_fig.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["crash_defense_drop_pct"],
                name="Local drop %",
                mode="lines",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            ),
            secondary_y=True,
        )
        detection_df = df[df["crash_defense_detected"] > 0]
        if not detection_df.empty:
            detection_hover = np.column_stack(
                (
                    detection_df["crash_defense_cash_spent"],
                    detection_df["crash_defense_buy_tokens"],
                    detection_df["crash_defense_gas_spend"],
                )
            )
            crash_fig.add_trace(
                go.Scatter(
                    x=detection_df["month"],
                    y=detection_df["crash_defense_drop_pct"],
                    mode="markers",
                    marker=dict(color="#c0392b", size=10, symbol="x"),
                    name="Crash detections",
                    customdata=detection_hover,
                    hovertemplate=
                        "Month %{x}<br>Drop %{y:.2f}%<br>Cash used %{customdata[0]:,.0f} USD"
                        "<br>Buys %{customdata[1]:,.2f} tokens"
                        "<br>Gas subsidies %{customdata[2]:,.0f} USD<extra></extra>",
                ),
                secondary_y=True,
            )
        max_drop = float(df.loc[crash_mask, "crash_defense_drop_pct"].max()) if crash_mask.any() else 0.0
        crash_fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Month",
            title_text="Crash defense interventions",
            barmode="stack",
        )
        crash_fig.update_yaxes(title_text="Token-equivalent support", secondary_y=False)
        crash_fig.update_yaxes(title_text="Detected drop (%)", secondary_y=True, range=[0, max(5.0, max_drop * 1.2 if max_drop else 5.0)])
        _plotly(crash_fig, full_width=True)
    else:
        st.caption("Crash defense mode never tripped during this simulation.")

    if not state_df.empty:
        st.divider()
        st.markdown("### Attacker analytics")
        pnl_fig = px.line(
            state_df,
            x="month",
            y="attacker_pnl",
            labels={"attacker_pnl": "PnL (USD)", "month": "Month"},
            template="plotly_white",
        )
        _plotly(pnl_fig, full_width=True)

        inv_fig = px.line(
            state_df,
            x="month",
            y=["attacker_cash", "attacker_tokens"],
            labels={"value": "Cash / Tokens", "variable": "Series"},
            template="plotly_white",
        )
        _plotly(inv_fig, full_width=True)

        auto_fig = px.bar(
            state_df,
            x="month",
            y="auto_flow_tokens",
            labels={"auto_flow_tokens": "Auto flow (tokens)", "month": "Month"},
            template="plotly_white",
        )
        _plotly(auto_fig, full_width=True)

    with st.expander("Raw timeline data"):
        st.dataframe(df, width="stretch")

    st.divider()
    st.markdown("### Attacker summary")
    final_pnl = None
    final_cash = None
    final_tokens = None
    if output.attacker_state is not None and not output.attacker_state.empty:
        final_row = output.attacker_state.iloc[-1]
        final_pnl = float(final_row.get("attacker_pnl", 0.0))
        final_cash = float(final_row.get("attacker_cash", 0.0))
        final_tokens = float(final_row.get("attacker_tokens", 0.0))

    if final_pnl is None and not output.attacker_trades.empty:
        final_price = df["token_price"].iloc[-1]
        cash = 0.0
        tokens = 0.0
        for row in output.attacker_trades.itertuples():
            if row.action == "buy":
                cash -= row.tokens * row.price
                tokens += row.tokens
            else:
                cash += row.tokens * row.price
                tokens -= row.tokens
        final_pnl = cash + tokens * final_price
        final_cash = cash
        final_tokens = tokens

    if final_pnl is None:
        st.info("No attacker trades recorded.")
    else:
        st.metric("Final attacker PnL (USD)", f"{final_pnl:,.0f}")
        if final_cash is not None and final_tokens is not None:
            st.caption(
                f"Ending cash ${final_cash:,.0f} | Ending tokens {final_tokens:,.0f}"
            )
        if not output.attacker_trades.empty:
            trade_df = output.attacker_trades.copy()
            if "month" not in trade_df.columns:
                trade_df["month"] = trade_df["step"] + 1
            trade_df["tokens_abs"] = trade_df["tokens"].abs()

            price_trace = px.line(
                df,
                x="month",
                y="token_price",
                labels={"token_price": "Price (USD)", "month": "Month"},
                template="plotly_white",
                title="Price with attacker trades",
            )
            trade_scatter = px.scatter(
                trade_df,
                x="month",
                y="price",
                color="action",
                size="tokens_abs",
                size_max=18,
                labels={"price": "Price (USD)", "action": "Action", "month": "Month", "tokens_abs": "|Tokens|"},
                hover_data={"tokens": True, "tag": True},
                template="plotly_white",
            )
            for trace in trade_scatter.data:
                price_trace.add_trace(trace)
            price_trace.update_layout(showlegend=True)
            _plotly(price_trace, full_width=True)

            st.dataframe(trade_df, width="stretch")


def render_market_conditions_tab(sim: SimulationConfig) -> SimulationConfig:
    st.subheader("🌍 Base Market Conditions")
    st.markdown(
        "This is where you set up the 'natural' behavior of your token's world—before any algorithms or defenses kick in."
    )
    st.info(
        "Tip: Run once with the defaults to establish a baseline. "
        "Then tweak one setting at a time to see cause and effect."
    )
    
    st.markdown("---")
    st.subheader("Market & Adoption")
    st.markdown(
        "How your token's market works: How long to simulate, how fast users join, how liquid the market is."
    )
    c1, c2 = st.columns(2)
    with c1:
        years = st.slider(
            "Simulation length (years)",
            1,
            15,
            sim.years,
            help="How many real-world years you want to watch play out. Stretching the horizon exposes slow-rolling failures that short runs miss.",
        )
        sim.years = years

        step_options = sorted({sim.steps_per_year, 12, 24, 48, 96, 144})
        steps_per_year = st.selectbox(
            "Time granularity",
            step_options,
            index=step_options.index(sim.steps_per_year),
            help="How often the model records a snapshot. Monthly (12) keeps charts readable; 48 or more unlocks higher-frequency attacker moves.",
        )
        sim.steps_per_year = steps_per_year

        sim.initial_gold_price = st.number_input(
            "Spot gold price (USD)",
            value=sim.initial_gold_price,
            step=25.0,
            help="Starting price of an ounce of gold in dollars. Think of it as the real-world anchor the token aspires to shadow.",
        )
        sim.gold_drift_annual = st.number_input(
            "Expected annual gold drift",
            value=sim.gold_drift_annual,
            step=0.005,
            format="%.4f",
            help="Average yearly change you expect for gold itself. 0.02 ≈ 2% annual growth, negative numbers mean a slow bleed.",
        )
        sim.gold_vol_annual = st.number_input(
            "Annual gold volatility",
            value=sim.gold_vol_annual,
            step=0.01,
            format="%.4f",
            help="How jumpy gold should be. Higher values let the anchor whip around, lower values keep the peg target calm.",
        )
        sim.impact_coeff = st.number_input(
            "Order-book thinness",
            value=sim.impact_coeff,
            step=0.02,
            format="%.2f",
            help="How fragile the market is to each trade. A higher number means one big order can shove price around; a lower number mimics deep liquidity.",
        )
    with c2:
        sim.initial_users = st.number_input(
            "Initial active users",
            value=int(sim.initial_users),
            step=250,
            help="Number of people already using the token on day zero. More users at launch means the system starts with built-in demand.",
        )
        sim.base_user_growth = st.number_input(
            "Base monthly user growth",
            value=sim.base_user_growth,
            step=0.001,
            format="%.4f",
            help="Core share of new users the project attracts each month without hype, price action, or campaigns.",
        )
        sim.growth_accel = st.number_input(
            "Growth acceleration",
            value=sim.growth_accel,
            step=0.0002,
            format="%.4f",
            help="How quickly that base growth snowballs as confidence builds. Higher numbers let word-of-mouth compound faster.",
        )
        sim.max_user_growth = st.number_input(
            "Ceiling monthly growth",
            value=sim.max_user_growth,
            step=0.005,
            format="%.4f",
            help="Upper limit on how much the user base can expand in a single month. It prevents runaway math from inventing impossible growth.",
        )
        sim.churn_rate = st.number_input(
            "Baseline monthly churn",
            value=sim.churn_rate,
            step=0.001,
            format="%.4f",
            help="Portion of users who wander off each month even when everything looks rosy. Every product has some natural churn.",
        )
        sim.organic_noise = st.number_input(
            "Organic flow noise",
            value=sim.organic_noise,
            step=0.005,
            format="%.4f",
            help="Random wobble in day-to-day trading. Bigger values make price action messier, like a token tossed around by headlines and influencers.",
        )

    with st.expander("Advanced market structure", expanded=False):
        st.caption("Dial in how liquidity and arbitrage behave at the base layer.")
        sim.hold_ratio_start = st.number_input(
            "Hold ratio at launch",
            value=sim.hold_ratio_start,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Share of early users who tuck tokens away instead of trading them. Raise it to model a loyal launch crowd.",
        )
        sim.hold_ratio_end = st.number_input(
            "Hold ratio at horizon",
            value=sim.hold_ratio_end,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Where you expect that hold share to land after a few years. Higher values mean the community matures into set-it-and-forget-it holders.",
        )
        sim.liquidity_base = st.number_input(
            "Base liquidity (USD)",
            value=sim.liquidity_base,
            step=100_000.0,
            help="Depth of the order book even if only the core team is trading. Think of it as the market maker budget at launch.",
        )
        sim.liquidity_per_user = st.number_input(
            "Liquidity per active user",
            value=sim.liquidity_per_user,
            step=5.0,
            help="How much fresh buying and selling power each new user brings. Big values assume adopters show up with capital.",
        )
        sim.baseline_reversion = st.number_input(
            "Baseline price reversion",
            value=sim.baseline_reversion,
            min_value=0.0,
            max_value=0.9,
            step=0.01,
            help="Strength of gravity that pulls price back toward fundamentals even without policy or attackers. Higher = quicker snap-back.",
        )
        sim.arbitrage_flow_strength = st.number_input(
            "Organic arbitrage flow strength",
            value=sim.arbitrage_flow_strength,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How forcefully regular traders lean into discounts or premiums. Crank it up to represent sharp arbitrage desks.",
        )
        st.caption("Capital cohorts let you shape who is funding demand at launch.")
        sim.retail_wallet_mean = st.number_input(
            "Average retail wallet (USD)",
            value=sim.retail_wallet_mean,
            step=100.0,
            help="Typical buying power per everyday user. Raising it assumes each person brings deeper pockets to the market.",
        )
        sim.retail_wallet_spread = st.number_input(
            "Retail wallet dispersion",
            value=sim.retail_wallet_spread,
            min_value=0.0,
            max_value=1.5,
            step=0.05,
            help="How unequal retail wallet sizes are. Higher values create a longer tail of users with bigger checks (and more volatility).",
        )
        sim.whale_user_fraction = st.number_input(
            "Whale user share",
            value=sim.whale_user_fraction,
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            format="%.2f",
            help="Slice of the user base that behaves like whales. They put serious money to work once confidence rises.",
        )
        sim.whale_balance_multiplier = st.number_input(
            "Whale balance multiplier",
            value=sim.whale_balance_multiplier,
            step=1.0,
            help="How much larger a whale wallet is relative to retail. Bigger multipliers mean a few whales can move the market sooner.",
        )
        sim.speculator_fraction = st.number_input(
            "Speculator fraction",
            value=sim.speculator_fraction,
            min_value=0.0,
            max_value=0.6,
            step=0.02,
            format="%.2f",
            help="Portion of users who chase momentum. They show up when hype builds but vanish fast if confidence slips.",
        )
        sim.speculator_base_balance = st.number_input(
            "Speculator wallet (USD)",
            value=sim.speculator_base_balance,
            step=100.0,
            help="Buying power per speculator when they are active. Pair it with the fraction above to simulate trader capital.",
        )
        sim.capital_noise = st.number_input(
            "Capital shock volatility",
            value=sim.capital_noise,
            min_value=0.0,
            max_value=0.6,
            step=0.02,
            format="%.2f",
            help="Random multiplier on wallet sizes each step. Higher numbers model fickle liquidity conditions and surprise capital inflows.",
        )
        sim.confidence_sensitivity = st.number_input(
            "Confidence sensitivity",
            value=sim.confidence_sensitivity,
            min_value=0.5,
            max_value=6.0,
            step=0.1,
            help="How emotional the community is when price wanders. Higher numbers exaggerate fear and greed swings.",
        )
        sim.treasury_seed_tokens = st.number_input(
            "Initial treasury tokens",
            value=sim.treasury_seed_tokens,
            step=10_000.0,
            help="Token inventory waiting in the treasury before any policy flips on. This pile is the ammo for future interventions.",
        )
        sim.treasury_seed_cash = st.number_input(
            "Initial treasury cash (USD)",
            value=sim.treasury_seed_cash,
            step=10_000.0,
            help="Dollars sitting in the treasury on day one. Cash lets the team defend the peg or subsidize traders before any fundraising.",
        )

    st.subheader("Intrinsic Pricing Drivers")
    d1, d2 = st.columns(2)
    with d1:
        sim.intrinsic_value_start = st.number_input(
            "Intrinsic base price",
            value=sim.intrinsic_value_start,
            step=50.0,
            help="The starting 'fair price' the model believes in before hype or policy help. When everything is calm, price gravitates back here.",
        )
        sim.intrinsic_growth_exponent = st.number_input(
            "Adoption price exponent",
            value=sim.intrinsic_growth_exponent,
            min_value=0.05,
            max_value=1.5,
            step=0.05,
            help="Controls how much extra value each new user adds. Higher numbers mean growth compounds fast; lower numbers keep the curve gentle.",
        )
        sim.intrinsic_noise = st.number_input(
            "Intrinsic drift volatility",
            value=sim.intrinsic_noise,
            min_value=0.0,
            max_value=0.5,
            step=0.005,
            format="%.3f",
            help="Random story beats that nudge the fair value around even when fundamentals are steady. Bigger values mimic rumor-driven projects.",
        )
        sim.hype_start = st.number_input(
            "Initial hype index",
            value=sim.hype_start,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How hot the narrative is at launch. Zero = nobody cares, 1+ = people are already talking about it.",
        )
        sim.hype_decay = st.number_input(
            "Hype decay rate",
            value=sim.hype_decay,
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            format="%.3f",
            help="How fast hype fades when nothing exciting happens. Larger numbers drain buzz quickly unless you feed the story.",
        )
    with d2:
        sim.intrinsic_confidence_weight = st.number_input(
            "Confidence weight",
            value=sim.intrinsic_confidence_weight,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How much community belief can bend the fair price. Crank it up if diamond hands and evangelists truly set the tone.",
        )
        sim.intrinsic_liquidity_weight = st.number_input(
            "Liquidity weight",
            value=sim.intrinsic_liquidity_weight,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How strongly deep order books translate into safety and higher fair value. Higher = liquidity matters a lot, lower = value ignores it.",
        )
        sim.gold_guidance_strength = st.number_input(
            "Soft gold narrative weight",
            value=sim.gold_guidance_strength,
            min_value=0.0,
            max_value=3.0,
            step=0.1,
            help="How loudly the community chants 'we should track gold'. Higher values tug the fair price toward gold even when no policy is running.",
        )
        sim.hype_sensitivity_growth = st.number_input(
            "Hype sensitivity (growth)",
            value=sim.hype_sensitivity_growth,
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            help="How quickly hype jumps when user numbers climb. High settings assume social proof kicks in hard.",
        )
        sim.hype_sensitivity_price = st.number_input(
            "Hype sensitivity (price momentum)",
            value=sim.hype_sensitivity_price,
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            help="How much price rallies (or dumps) fuel the narrative. Bigger numbers let short squeezes ignite a runaway storyline.",
        )

    st.subheader("Adoption Dynamics")
    ad1, ad2 = st.columns(2)
    with ad1:
        sim.user_carrying_capacity = st.number_input(
            "Carrying capacity (users)",
            value=int(sim.user_carrying_capacity),
            min_value=10_000,
            step=25_000,
            help="Rough cap on how many users you think exist in the addressable market. The model slows down as you approach this ceiling.",
        )
        sim.adoption_saturation_power = st.number_input(
            "Saturation curve power",
            value=sim.adoption_saturation_power,
            min_value=0.5,
            max_value=2.5,
            step=0.05,
            help="Shapes how gently growth eases off near saturation. Bigger numbers create a steep slowdown once you approach the total market.",
        )
        sim.adoption_volatility = st.number_input(
            "Adoption noise",
            value=sim.adoption_volatility,
            min_value=0.0,
            max_value=0.1,
            step=0.001,
            format="%.3f",
            help="How choppy user signups should be. Raise it to mimic months where news drives spikes and lulls.",
        )
        sim.macro_shock_chance = st.number_input(
            "Macro shock chance (annual)",
            value=sim.macro_shock_chance,
            min_value=0.0,
            max_value=0.25,
            step=0.005,
            format="%.3f",
            help="Chance each year that a nasty outside event (regulation, hacks, wars) happens and rattles adoption.",
        )
    with ad2:
        sim.adoption_price_sensitivity = st.number_input(
            "Hype sensitivity",
            value=sim.adoption_price_sensitivity,
            min_value=0.0,
            max_value=1.5,
            step=0.05,
            help="How quickly buzz spreads when the token is doing well. Higher numbers mean people pile in when they see gains.",
        )
        sim.churn_price_sensitivity = st.number_input(
            "Pain sensitivity",
            value=sim.churn_price_sensitivity,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How brutal price drops feel to casual holders. Crank it up and dips shove people to the exits fast.",
        )
        sim.churn_confidence_weight = st.number_input(
            "Confidence weight on churn",
            value=sim.churn_confidence_weight,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="How much shaky vibes translate into actual user loss. Higher values connect sentiment directly to churn.",
        )
        sim.macro_shock_magnitude = st.number_input(
            "Macro shock severity",
            value=sim.macro_shock_magnitude,
            min_value=0.05,
            max_value=0.6,
            step=0.01,
            help="Size of the hit when a macro shock shows up. Bigger values mean the nasty event wipes out users and confidence more aggressively.",
        )

    st.subheader("Token Supply Controls")
    plan = sim.supply_plan
    for attr, default_value in {
        "absolute_supply_enabled": False,
        "absolute_supply_tokens": None,
    }.items():
        if not hasattr(plan, attr):
            setattr(plan, attr, default_value)

    if not hasattr(plan, "absolute_supply_enabled"):
        setattr(plan, "absolute_supply_enabled", False)
    if not hasattr(plan, "absolute_supply_tokens"):
        setattr(plan, "absolute_supply_tokens", None)

    total_launch_supply = plan.initial_circulating + plan.initial_locked + plan.initial_treasury
    inferred_mode = "managed"
    if plan.absolute_supply_enabled:
        inferred_mode = "absolute"
    elif plan.supply_regime == "hard_cap" and plan.baseline_emission_per_user == 0.0 and plan.goal_emission_per_user == 0.0:
        inferred_mode = "flat"
    supply_mode_labels = {
        "managed": "Manual (use detailed sliders)",
        "flat": "Flat supply (no emissions)",
        "expansion": "Linear expansion",
        "absolute": "Absolute override",
    }
    supply_mode = st.selectbox(
        "Supply mode",
        list(supply_mode_labels.keys()),
        index=list(supply_mode_labels.keys()).index(inferred_mode),
        format_func=lambda key: supply_mode_labels[key],
        help="Pick a preset for supply behaviour or stay in manual mode for full control.",
    )
    if supply_mode == "flat":
        plan.absolute_supply_enabled = False
        plan.supply_regime = "hard_cap"
        plan.baseline_emission_per_user = 0.0
        plan.goal_emission_per_user = 0.0
    elif supply_mode == "expansion":
        plan.absolute_supply_enabled = False
        plan.supply_regime = "balanced"
        exp_cols = st.columns(2)
        with exp_cols[0]:
            annual_expansion_pct = st.slider(
                "Target annual supply growth (%)",
                min_value=-10.0,
                max_value=60.0,
                value=10.0,
                step=1.0,
                help="Desired percentage change in circulating supply per year. Negative values slowly shrink the float.",
            )
        with exp_cols[1]:
            plan.goal_release_multiplier = st.slider(
                "Expansion aggressiveness",
                min_value=0.5,
                max_value=6.0,
                value=float(plan.goal_release_multiplier),
                step=0.1,
                help="Multiplier that governs how fast rewards scale when goals are met.",
            )
        monthly_rate = (1 + annual_expansion_pct / 100.0) ** (1 / max(sim.steps_per_year, 1)) - 1
        plan.baseline_emission_per_user = max(0.0, monthly_rate * plan.initial_circulating / max(sim.initial_users, 1))
        plan.goal_emission_per_user = plan.baseline_emission_per_user * plan.goal_release_multiplier
    elif supply_mode == "absolute":
        plan.absolute_supply_enabled = True
        default_abs = plan.absolute_supply_tokens or total_launch_supply
        plan.absolute_supply_tokens = st.number_input(
            "Absolute supply target",
            value=float(default_abs),
            step=100_000.0,
            help="Total tokens the system enforces each step. Minting/unlocking is overridden to stay near this level.",
        )
    else:
        plan.absolute_supply_enabled = False
    sup1, sup2 = st.columns(2)
    with sup1:
        plan.initial_circulating = st.number_input(
            "Launch circulating supply",
            value=float(plan.initial_circulating),
            step=50_000.0,
            help="Tradable tokens on day zero. Set this tiny to model a super-limited float at launch.",
        )
        plan.initial_locked = st.number_input(
            "Locked reserve (vesting)",
            value=float(plan.initial_locked),
            step=500_000.0,
            help="Tokens that exist but can't trade yet. They unlock slowly based on the schedule below.",
        )
        plan.initial_treasury = st.number_input(
            "Treasury seed tokens",
            value=float(plan.initial_treasury),
            step=100_000.0,
            help="Inventory sitting in the treasury before interventions. Used for future market operations.",
        )
        plan.launch_float_release = st.slider(
            "Launch unlock fraction",
            min_value=0.0,
            max_value=0.5,
            value=float(plan.launch_float_release),
            step=0.01,
            help="Slice of the locked reserve you drip into the market immediately after launch.",
        )
        sim.supply_floor = st.number_input(
            "Circulating supply floor",
            value=float(sim.supply_floor),
            step=250_000.0,
            help="Safety rail that prevents burns from deleting the entire float.",
        )
    with sup2:
        plan.goal_price = st.number_input(
            "Goal price trigger (USD)",
            value=float(plan.goal_price),
            step=25.0,
            help="Once the token clears this price, emissions ramp toward the goal schedule.",
        )
        plan.goal_release_multiplier = st.slider(
            "Reward boost after goal",
            min_value=0.5,
            max_value=6.0,
            value=float(plan.goal_release_multiplier),
            step=0.1,
            help="How aggressively rewards expand once the goal price is hit.",
        )
        plan.baseline_emission_per_user = st.number_input(
            "Baseline emission / user",
            value=float(plan.baseline_emission_per_user),
            min_value=0.0,
            step=0.05,
            format="%.3f",
            help="Reward budget for each new user before the goal price is reached.",
        )
        plan.goal_emission_per_user = st.number_input(
            "Goal emission / user",
            value=float(plan.goal_emission_per_user),
            min_value=0.0,
            step=0.1,
            format="%.3f",
            help="Steady-state reward budget once the project is cruising above goal.",
        )
        plan.supply_hard_cap = st.number_input(
            "Hard supply cap",
            value=float(plan.supply_hard_cap),
            step=5_000_000.0,
            help="Absolute upper bound on total tokens the model is allowed to mint.",
        )

    with st.expander("Advanced supply mechanics", expanded=False):
        regime_options = ["balanced", "decay", "hard_cap", "adaptive"]
        current_regime = plan.supply_regime if plan.supply_regime in regime_options else "balanced"
        plan.supply_regime = st.selectbox(
            "Supply regime",
            regime_options,
            index=regime_options.index(current_regime),
            help="Balanced keeps the existing behaviour. Decay tapers emissions over time, Hard cap enforces a strict ceiling, Adaptive scales emissions with price bands.",
        )
        adv1, adv2 = st.columns(2)
        with adv1:
            plan.halving_interval_steps = st.number_input(
                "Halving interval (steps)",
                value=int(plan.halving_interval_steps),
                min_value=1,
                step=1,
                help="How many simulation steps before the reward target is halved after the goal is hit.",
            )
            plan.halving_factor = st.slider(
                "Halving factor",
                min_value=0.1,
                max_value=1.0,
                value=float(plan.halving_factor),
                step=0.05,
                help="Multiplier applied to emissions at each halving. 0.5 mimics bitcoin-style cuts.",
            )
            plan.unlock_slope = st.number_input(
                "Unlock slope",
                value=float(plan.unlock_slope),
                min_value=0.0,
                step=0.001,
                format="%.4f",
                help="Portion of the locked reserve that drips out each step once price momentum cooperates.",
            )
            plan.unlock_jitter = st.slider(
                "Unlock variability",
                min_value=0.0,
                max_value=1.0,
                value=float(plan.unlock_jitter),
                step=0.05,
                help="Randomizes unlock amounts to keep supply charts from looking robotic.",
            )
            plan.inflation_guard_price = st.number_input(
                "Inflation guard price multiple",
                value=float(plan.inflation_guard_price),
                min_value=0.0,
                step=0.05,
                help="If price falls below this multiple of NAV the mint faucet throttles hard.",
            )
            plan.inflation_guard_cooldown = st.number_input(
                "Inflation guard cooldown (steps)",
                value=int(plan.inflation_guard_cooldown),
                min_value=0,
                step=1,
                help="How long the faucet stays throttled after the guard trips.",
            )
            if plan.supply_regime == "decay":
                plan.decay_half_life_years = st.number_input(
                    "Emission half-life (years)",
                    value=float(plan.decay_half_life_years),
                    min_value=0.1,
                    step=0.1,
                    help="Number of years before emissions halve under the decay regime.",
                )
        with adv2:
            plan.tx_tax_rate = st.number_input(
                "Transaction tax rate",
                value=float(plan.tx_tax_rate),
                min_value=0.0,
                max_value=0.25,
                step=0.001,
                format="%.3f",
                help="Portion of traded tokens siphoned off each step as a burn/tax.",
            )
            plan.burn_share_of_tax = st.slider(
                "Tax burned immediately",
                min_value=0.0,
                max_value=1.0,
                value=float(plan.burn_share_of_tax),
                step=0.05,
                help="Share of the transaction tax that disappears instantly.",
            )
            plan.burn_vault_share = st.slider(
                "Tax routed to burn vault",
                min_value=0.0,
                max_value=1.0,
                value=float(plan.burn_vault_share),
                step=0.05,
                help="Share of the tax saved for emergency liquidity. Anything left over is ignored.",
            )
            plan.burn_vault_release_fraction = st.slider(
                "Vault release fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(plan.burn_vault_release_fraction),
                step=0.05,
                help="How much of the vault balance is returned to float when price falls under the release trigger.",
            )
            plan.burn_vault_release_threshold = st.number_input(
                "Vault release trigger (×goal)",
                value=float(plan.burn_vault_release_threshold),
                min_value=0.1,
                step=0.05,
                help="When price drops below this share of the goal price, the vault starts drip-feeding supply back in.",
            )
            plan.burn_pause_price = st.number_input(
                "Burn pause threshold (×NAV)",
                value=float(plan.burn_pause_price),
                min_value=0.0,
                step=0.05,
                help="Below this multiple of NAV the tax-driven burns pause so you stop starving the market.",
            )
            plan.burn_escalate_price = st.number_input(
                "Burn escalate threshold (×NAV)",
                value=float(plan.burn_escalate_price),
                min_value=0.5,
                step=0.05,
                help="Above this multiple of NAV the model burns extra tokens to squash runaway premiums.",
            )
            if plan.supply_regime == "adaptive":
                plan.adaptive_floor_price = st.number_input(
                    "Adaptive floor price (USD)",
                    value=float(plan.adaptive_floor_price),
                    step=10.0,
                    help="Below this price emissions taper to preserve scarcity.",
                )
                plan.adaptive_ceiling_price = st.number_input(
                    "Adaptive ceiling price (USD)",
                    value=float(plan.adaptive_ceiling_price),
                    step=10.0,
                    help="Above this level emissions accelerate to discourage overheating.",
                )
            if plan.supply_regime == "hard_cap":
                st.caption("Hard cap mode uses the supply hard cap and ignores emission boosts once the cap is hit.")

        trigger_cols = st.columns(3)
        with trigger_cols[0]:
            plan.price_trigger_unlock = st.number_input(
                "Unlock trigger price",
                value=float(plan.price_trigger_unlock or 0.0),
                min_value=0.0,
                step=10.0,
                help="When price crosses this level, additional locked tokens unlock instantly.",
            )
            plan.price_trigger_unlock_fraction = st.slider(
                "Unlock fraction",
                min_value=0.0,
                max_value=0.6,
                value=float(plan.price_trigger_unlock_fraction),
                step=0.02,
            )
        with trigger_cols[1]:
            plan.price_trigger_burn = st.number_input(
                "Burn trigger price",
                value=float(plan.price_trigger_burn or 0.0),
                min_value=0.0,
                step=10.0,
                help="Above this level the system opportunistically burns extra float.",
            )
            plan.price_trigger_burn_fraction = st.slider(
                "Burn fraction",
                min_value=0.0,
                max_value=0.2,
                value=float(plan.price_trigger_burn_fraction),
                step=0.01,
            )
        with trigger_cols[2]:
            plan.burn_floor_supply = st.number_input(
                "Burn floor supply",
                value=float(plan.burn_floor_supply),
                min_value=0.0,
                step=100_000.0,
                help="Never burn tokens if doing so would drop the float below this level.",
            )
            plan.supply_reversal_price = st.number_input(
                "Relock trigger price",
                value=float(plan.supply_reversal_price or 0.0),
                min_value=0.0,
                step=10.0,
                help="If price dives below this level, a slice of float can relock to rebuild scarcity.",
            )
            plan.supply_reversal_fraction = st.slider(
                "Relock fraction",
                min_value=0.0,
                max_value=0.6,
                value=float(plan.supply_reversal_fraction),
                step=0.02,
            )

    # Keep legacy knobs in sync for downstream compatibility
    sim.initial_free_float = float(plan.initial_circulating)
    sim.founder_locked = float(plan.initial_locked)
    sim.initial_treasury_tokens = float(plan.initial_treasury)

    st.subheader("Narrative & Hype")
    hype_cfg = sim.hype_settings
    hype_cols = st.columns(2)
    with hype_cols[0]:
        hype_cfg.enable_background_hype = st.checkbox(
            "Enable background hype events",
            value=hype_cfg.enable_background_hype,
            help="When on, social buzz randomly triggers user stampedes that feed the hype index.",
        )
        hype_cfg.random_hype_chance = st.number_input(
            "Annual chance of organic hype",
            value=float(hype_cfg.random_hype_chance),
            min_value=0.0,
            max_value=0.6,
            step=0.01,
            format="%.2f",
            help="Rough probability (per year) that a new hype cycle appears out of nowhere.",
        )
        hype_cfg.min_duration_steps = st.number_input(
            "Minimum hype duration (steps)",
            value=int(hype_cfg.min_duration_steps),
            min_value=1,
            step=1,
        )
        hype_cfg.max_duration_steps = st.number_input(
            "Maximum hype duration (steps)",
            value=int(hype_cfg.max_duration_steps),
            min_value=1,
            step=1,
        )
        hype_cfg.base_intensity = st.slider(
            "Base hype intensity",
            min_value=0.0,
            max_value=1.5,
            value=float(hype_cfg.base_intensity),
            step=0.05,
            help="Baseline buying pressure generated by each hype pulse.",
        )
    with hype_cols[1]:
        hype_cfg.viral_spike_chance = st.slider(
            "Chance hype goes viral",
            min_value=0.0,
            max_value=0.3,
            value=float(hype_cfg.viral_spike_chance),
            step=0.01,
            help="Probability a hype event explodes into a viral wave instead of a mild bump.",
        )
        hype_cfg.viral_intensity_multiplier = st.slider(
            "Viral intensity multiplier",
            min_value=1.0,
            max_value=4.0,
            value=float(hype_cfg.viral_intensity_multiplier),
            step=0.1,
            help="How much stronger a viral spike hits compared with a normal hype pulse.",
        )
        hype_cfg.whale_bias = st.slider(
            "Whale bias in hype cohorts",
            min_value=0.0,
            max_value=1.0,
            value=float(hype_cfg.whale_bias),
            step=0.05,
            help="Higher values mean hype waves attract disproportionately more whales.",
        )
        hype_cfg.quick_flip_ratio = st.slider(
            "Quick-flip share",
            min_value=0.0,
            max_value=1.0,
            value=float(hype_cfg.quick_flip_ratio),
            step=0.05,
            help="Portion of hype entrants who dump shortly after buying.",
        )
        hype_cfg.retention_decay = st.slider(
            "Hype retention decay",
            min_value=0.0,
            max_value=1.0,
            value=float(hype_cfg.retention_decay),
            step=0.05,
            help="Speed at which hype tourists exit once the story cools off.",
        )
        hype_cfg.hype_cooldown_steps = st.number_input(
            "Cooldown between hype events",
            value=int(hype_cfg.hype_cooldown_steps),
            min_value=0,
            step=1,
            help="Minimum wait (in steps) before another hype wave can spark.",
        )

    return sim


def render_algorithm_tab(sim: SimulationConfig, policy: PolicySettings) -> Tuple[SimulationConfig, PolicySettings]:
    st.subheader("🛠️ Algorithm Forge")
    st.markdown(
        "Design your automated defense systems. These kick in when the peg starts to wobble and try to stabilize things."
    )
    st.info(
        "Tip: For a first comparison, keep modules disabled to observe the natural market. "
        "Then enable the Algorithm in 'defend' mode and compare results."
    )
    st.markdown("---")
    algo_cfg = sim.algorithm_settings
    algo_cols = st.columns(3)
    with algo_cols[0]:
        algo_cfg.enabled = st.checkbox(
            "Enable algorithm",
            value=algo_cfg.enabled,
            help=(
                "Turns on the autonomous treasury brain. When enabled, every objective you set below feeds live signals into the simulator—even if you leave older manual modules off. "
                "Think of it as the autopilot for minting, burning, and liquidity moves."
            ),
        )
    with algo_cols[1]:
        algo_cfg.master_switch = st.checkbox(
            "Master override",
            value=algo_cfg.master_switch,
            help=(
                "Forces the algorithm to stay active even when the legacy peg-defense treasury is toggled off. "
                "Use this when you want to compare the pure AI strategy against a no-intervention baseline."
            ),
        )
    with algo_cols[2]:
        mode_options = ["defend", "grow", "custom", "stabilize"]
        current_mode = algo_cfg.mode if algo_cfg.mode in mode_options else "defend"
        algo_cfg.mode = st.selectbox(
            "Playbook mode",
            mode_options,
            index=mode_options.index(current_mode),
            help=(
                "High-level vibe for preset defaults: "
                "• Defend = cling to NAV, "
                "• Grow = lean hard into user expansion and price appreciation, "
                "• Stabilize = focus on low volatility, "
                "• Custom = you hand-pick everything below."
            ),
        )
    prev_mode = st.session_state.get("algo_playbook_prev_mode")
    if prev_mode is None:
        st.session_state["algo_playbook_prev_mode"] = algo_cfg.mode
    elif algo_cfg.mode != prev_mode:
        if algo_cfg.mode != "custom":
            _apply_playbook_presets(algo_cfg, algo_cfg.mode)
            st.success(f"Applied the {algo_cfg.mode} preset.")
        st.session_state["algo_playbook_prev_mode"] = algo_cfg.mode

    algo_meta_cols = st.columns(3)
    with algo_meta_cols[0]:
        algo_cfg.discretionary_budget = st.number_input(
            "Discretionary budget (USD)",
            value=float(algo_cfg.discretionary_budget),
            step=50_000.0,
            help=(
                "Extra USD pile the bot can burn through before the main treasury reserves jump in. "
                "Great for gentle market-making nudges: e.g. set 250,000 to cover early volatility without risking core reserves."
            ),
        )
    with algo_meta_cols[1]:
        algo_cfg.treasury_ramp_years = st.number_input(
            "Treasury ramp-in (years)",
            value=float(algo_cfg.treasury_ramp_years),
            min_value=0.0,
            step=0.1,
            help=(
                "How long the algorithm takes to gain full authority over treasury firepower. "
                "0 = instant control, 2.0 = it eases in over two simulation years with progressively larger interventions."
            ),
        )
    with algo_meta_cols[2]:
        algo_cfg.goal_note = st.text_input(
            "Strategy note",
            value=algo_cfg.goal_note,
            help="Optional annotation stored with scenario outputs.",
        )

    st.markdown("#### Objective Builder")
    st.caption(
        "Pick the levers the algorithm should care about. Each block below turns into a live objective with clearer wording, "
        "so you always know if a number is in USD, %, or raw units."
    )

    steps_per_year = max(int(sim.steps_per_year), 1)
    default_mode = algo_cfg.mode
    recognisable_metrics = {"peg_deviation", "growth", "price", "volatility", "treasury_nav"}
    primary_objectives: Dict[str, AlgorithmObjective] = {}
    advanced_objectives: List[AlgorithmObjective] = []
    for objective in algo_cfg.objectives or []:
        metric_key = getattr(objective, "metric", "")
        if metric_key in recognisable_metrics and metric_key not in primary_objectives:
            primary_objectives[metric_key] = objective
        else:
            advanced_objectives.append(objective)

    curated_objectives: List[AlgorithmObjective] = []

    # Peg stability objective
    peg_existing = primary_objectives.get("peg_deviation")
    peg_enabled_default = (peg_existing.enabled if peg_existing else False) or default_mode in ("defend", "stabilize")
    peg_container = st.container()
    with peg_container:
        peg_enabled = st.checkbox(
            "Hold the peg tight (peg deviation)",
            value=peg_enabled_default,
            help=(
                "Keeps market price hugging NAV (gold/fundamental anchors). Tolerance is a ±% band around the peg. "
                "Example: 4% tolerance means the bot stays calm until price drifts 4% away, then ramps interventions."
            ),
        )
        if peg_enabled:
            peg_tolerance_pct = st.slider(
                "Comfort band (percent away from NAV)",
                min_value=1.0,
                max_value=15.0,
                value=float((peg_existing.tolerance if peg_existing else 0.04) * 100.0),
                step=0.5,
                help=(
                    "How much deviation you are OK with before the bot leans in. Values are symmetric around zero. "
                    "2% = extremely tight, 10% = allow swings before acting."
                ),
            )
            peg_weight = st.slider(
                "Intervention weight",
                min_value=0.2,
                max_value=3.0,
                value=float(peg_existing.weight if peg_existing else 1.2),
                step=0.1,
                help=(
                    "Heavier weight = the peg objective outranks other goals when signals conflict. "
                    "Use 1.5–2.0 when defending under heavy sell pressure."
                ),
            )
            peg_horizon = st.slider(
                "Look-ahead horizon (years)",
                min_value=0.1,
                max_value=3.0,
                value=float(peg_existing.horizon_years if peg_existing else 0.6),
                step=0.1,
                help=(
                    "How far ahead the algorithm smooths peg deviations. Short horizons = faster, twitchier responses. "
                    "Long horizons = slower, steadier defence."
                ),
            )
            curated_objectives.append(
                AlgorithmObjective(
                    name="Peg Guard Rail",
                    metric="peg_deviation",
                    target_value=0.0,
                    tolerance=peg_tolerance_pct / 100.0,
                    comparison="track",
                    weight=peg_weight,
                    horizon_years=peg_horizon,
                    enabled=True,
                )
            )

    st.markdown("---")

    # Growth focus objective
    growth_existing = primary_objectives.get("growth")
    growth_enabled_default = (growth_existing.enabled if growth_existing else False) or default_mode == "grow"
    growth_enabled = st.checkbox(
        "Accelerate user growth",
        value=growth_enabled_default,
        help=(
            "Targets adoption_rate (new users per step). Inputs below are annualized so you think in normal business terms. "
            "Set a target you want to hit and whether the bot should simply exceed it or push the gas to maximise growth."
        ),
    )
    if growth_enabled:
        existing_growth_annual = _step_to_annual_rate(growth_existing.target_value, steps_per_year) if growth_existing else 0.45
        growth_target_pct = st.slider(
            "Desired annual user growth",
            min_value=10.0,
            max_value=400.0,
            value=float(existing_growth_annual * 100.0),
            step=5.0,
            help=(
                "Annual compound user growth rate. 60% ≈ 4% per month at 12 steps/year; 200% ≈ 2.2× users year-over-year. "
                "This drives how aggressively the algorithm feeds hype, marketing, and incentive mechanics."
            ),
        )
        existing_growth_tol_annual = _step_to_annual_rate(
            growth_existing.tolerance if growth_existing else 0.08, steps_per_year
        )
        growth_tolerance_pct = st.slider(
            "Slack around the target (annualised)",
            min_value=5.0,
            max_value=150.0,
            value=float(existing_growth_tol_annual * 100.0),
            step=5.0,
            help=(
                "Allowable gap between actual adoption and the target before the bot changes behaviour. "
                "Bigger values prevent overreactions when growth naturally oscillates."
            ),
        )
        growth_goal_style = st.selectbox(
            "Goal style",
            options=["maximize", "at_least"],
            index=0 if (growth_existing and growth_existing.comparison == "maximize") else 1,
            format_func=lambda key: "Push to maximize growth" if key == "maximize" else "Stay above the target rate",
            help=(
                "• Push to maximize growth: the bot keeps spending firepower as long as growth is rising.\n"
                "• Stay above the target rate: the bot accelerates only when growth dips below the floor."
            ),
        )
        growth_weight = st.slider(
            "Growth priority weight",
            min_value=0.2,
            max_value=3.0,
            value=float(growth_existing.weight if growth_existing else 1.0),
            step=0.1,
            help=(
                "If you crank this above your peg weight, the algorithm will happily trade short-term volatility for user growth. "
                "Lower weights let stability win in a tie."
            ),
        )
        growth_horizon = st.slider(
            "Growth look-ahead (years)",
            min_value=0.25,
            max_value=5.0,
            value=float(growth_existing.horizon_years if growth_existing else 1.5),
            step=0.25,
            help=(
                "Longer horizons mean the bot is patient and will judge growth on multi-year arcs. "
                "Short horizons force it to intervene if momentum fades for a few months."
            ),
        )
        curated_objectives.append(
            AlgorithmObjective(
                name="Growth Accelerator",
                metric="growth",
                target_value=_annual_to_step_rate(growth_target_pct / 100.0, steps_per_year),
                tolerance=_annual_to_step_rate(growth_tolerance_pct / 100.0, steps_per_year),
                comparison=growth_goal_style,
                weight=growth_weight,
                horizon_years=growth_horizon,
                enabled=True,
            )
        )

    st.markdown("---")

    # Treasury NAV / health objective
    nav_existing = primary_objectives.get("treasury_nav")
    nav_enabled_default = nav_existing.enabled if nav_existing else default_mode in ("defend", "custom")
    nav_enabled = st.checkbox(
        "Keep treasury backing rich",
        value=nav_enabled_default,
        help=(
            "Watches the gap between NAV and market price to ensure reserves stay healthy. "
            "Useful when you want the bot to buy aggressively while the token trades below backing."
        ),
    )
    if nav_enabled:
        nav_tolerance_pct = st.slider(
            "Treasury gap tolerance (percent)",
            min_value=2.0,
            max_value=40.0,
            value=float((nav_existing.tolerance if nav_existing else 0.10) * 100.0),
            step=1.0,
            help=(
                "How much undervaluation (market below NAV) triggers emergency interventions. "
                "Example: 20% tolerance lets price sit 20% under NAV before the bot empties the treasury."
            ),
        )
        nav_weight = st.slider(
            "Treasury objective weight",
            min_value=0.2,
            max_value=3.0,
            value=float(nav_existing.weight if nav_existing else 0.9),
            step=0.1,
            help=(
                "Higher weight = treasury health trumps other objectives. Handy when building cash/token buffers matters more than price smoothness."
            ),
        )
        nav_goal_style = st.selectbox(
            "Treasury goal style",
            options=["at_least", "maximize"],
            index=0 if (nav_existing and nav_existing.comparison == "at_least") else 1,
            format_func=lambda key: "Close the gap (at least break-even)" if key == "at_least" else "Stuff the treasury (maximize margin)",
            help=(
                "Choose between simply defending NAV (at least break-even) or chasing excess reserves (maximize). "
                "Maximize is more aggressive and may sacrifice peg smoothness."
            ),
        )
        nav_horizon = st.slider(
            "Treasury look-ahead (years)",
            min_value=0.25,
            max_value=5.0,
            value=float(nav_existing.horizon_years if nav_existing else 1.0),
            step=0.25,
            help="Long horizons let the bot rebuild reserves slowly; short horizons trigger quicker treasury actions.",
        )
        curated_objectives.append(
            AlgorithmObjective(
                name="Treasury Cushion",
                metric="treasury_nav",
                target_value=0.0,
                tolerance=nav_tolerance_pct / 100.0,
                comparison=nav_goal_style,
                weight=nav_weight,
                horizon_years=nav_horizon,
                enabled=True,
            )
        )

    st.markdown("---")

    # Volatility control
    vol_existing = primary_objectives.get("volatility")
    vol_enabled = st.checkbox(
        "Dampen volatility",
        value=bool(vol_existing.enabled) if vol_existing else default_mode == "stabilize",
        help=(
            "Targets realised volatility (standard deviation of price moves). "
            "Keeps swings calmer at the cost of slower responses to real shifts."
        ),
    )
    if vol_enabled:
        vol_tolerance_pct = st.slider(
            "Max acceptable realised volatility (percent)",
            min_value=5.0,
            max_value=150.0,
            value=float((_step_to_annual_rate(vol_existing.tolerance, steps_per_year) if vol_existing else 0.45) * 100.0),
            step=5.0,
            help=(
                "Upper bound on annualised price volatility before the bot steps in. "
                "Example: 60% ≈ 5% swings per month if you run 12 steps per year."
            ),
        )
        vol_weight = st.slider(
            "Volatility weight",
            min_value=0.2,
            max_value=3.0,
            value=float(vol_existing.weight if vol_existing else 0.8),
            step=0.1,
            help="Raise this when you want ultra-smooth price action even if it drifts from NAV or growth slows.",
        )
        vol_horizon = st.slider(
            "Volatility horizon (years)",
            min_value=0.25,
            max_value=5.0,
            value=float(vol_existing.horizon_years if vol_existing else 1.2),
            step=0.25,
            help="Longer horizons average volatility over more history. Short horizons chase every wiggle.",
        )
        curated_objectives.append(
            AlgorithmObjective(
                name="Calm Seas",
                metric="volatility",
                target_value=0.0,
                tolerance=_annual_to_step_rate(vol_tolerance_pct / 100.0, steps_per_year),
                comparison="at_most",
                weight=vol_weight,
                horizon_years=vol_horizon,
                enabled=True,
            )
        )

    st.markdown("---")

    # Explicit price target objective
    price_existing = primary_objectives.get("price")
    price_enabled = st.checkbox(
        "Aim for a specific price",
        value=bool(price_existing.enabled) if price_existing else False,
        help=(
            "Great for growth scenarios where you want valuation marching upward. "
            "Use with caution—if you ignore peg deviation, the token can drift far from fundamentals."
        ),
    )
    if price_enabled:
        price_target = st.number_input(
            "Price target (USD)",
            value=float(price_existing.target_value if price_existing else max(sim.algorithm_settings.custom_params.get("last_price_target", 0.0), 1_500.0)),
            min_value=0.0,
            step=50.0,
            help=(
                "Absolute market price the bot tries to sit on. Example: 2,000 = chase a $2k token. "
                "Keep this aligned with your emission plan or you'll mint/burn relentlessly."
            ),
        )
        price_tolerance = st.number_input(
            "Tolerance (USD)",
            value=float(price_existing.tolerance if price_existing else 150.0),
            min_value=0.0,
            step=10.0,
            help=(
                "Acceptable window around the target. "
                "Example: target $2,000 with $150 tolerance → bot tries to stay between $1,850 and $2,150."
            ),
        )
        price_comparison_options = ["track", "at_least", "at_most"]
        raw_price_comparison = price_existing.comparison if price_existing else "track"
        if raw_price_comparison == "maximize":
            raw_price_comparison = "at_least"
        elif raw_price_comparison == "minimize":
            raw_price_comparison = "at_most"
        if raw_price_comparison not in price_comparison_options:
            raw_price_comparison = "track"
        price_comparison = st.selectbox(
            "Price behaviour",
            options=price_comparison_options,
            index=price_comparison_options.index(raw_price_comparison),
            help=(
                "Track = pin the price to the target, At least = focus on never dropping below, "
                "At most = cap runaway rallies."
            ),
        )
        price_weight = st.slider(
            "Price target weight",
            min_value=0.2,
            max_value=3.0,
            value=float(price_existing.weight if price_existing else 0.9),
            step=0.1,
            help="Higher numbers give the price target veto power over other goals.",
        )
        price_horizon = st.slider(
            "Price horizon (years)",
            min_value=0.25,
            max_value=5.0,
            value=float(price_existing.horizon_years if price_existing else 1.0),
            step=0.25,
            help="Short horizons chase the target day-to-day. Long horizons nudge price there gradually.",
        )
        curated_objectives.append(
            AlgorithmObjective(
                name="Price Target",
                metric="price",
                target_value=price_target,
                tolerance=price_tolerance,
                comparison=price_comparison,
                weight=price_weight,
                horizon_years=price_horizon,
                enabled=True,
            )
        )
        algo_cfg.custom_params["last_price_target"] = price_target

    st.markdown("#### Growth Treasury Autopilot")
    st.caption(
        "Let the treasury watch its cash flow and auto-toggle fees. When coffers swell it redeploys into buybacks; "
        "when cash bleeds it cranks fees and eases off the gas so your growth push doesn't bankrupt the vault."
    )
    base_tax_pct = float(getattr(sim.supply_plan, "tx_tax_rate", 0.012) * 100.0)
    autopilot_cfg = dict(algo_cfg.custom_params.get("growth_autopilot", {}))
    auto_default = default_mode == "grow"
    growth_oracle_accuracy = float(autopilot_cfg.get("oracle_accuracy", 0.5))
    focus_labels = {
        "market_cap": "maximize market cap",
        "token_price": "maximize token price",
        "hybrid": "hybrid growth (cap + price)",
    }
    focus_keys = list(focus_labels.keys())
    growth_oracle_goal = str(autopilot_cfg.get("target_metric", "market_cap"))
    focus_index = focus_keys.index(growth_oracle_goal if growth_oracle_goal in focus_keys else "market_cap")
    growth_oracle_goal = st.selectbox(
        "Primary objective focus",
        focus_keys,
        index=focus_index,
        format_func=lambda key: focus_labels.get(key, key),
        help="Signals, autopilot heuristics, and oracle tweaks aim at this headline goal.",
    )
    algo_cfg.custom_params["primary_goal_mode"] = growth_oracle_goal

    growth_auto_enabled = st.checkbox(
        "Enable Growth Treasury Autopilot",
        value=bool(autopilot_cfg.get("enabled", auto_default)),
        help=(
            "When on, the sim tracks treasury cash each step. If cash is shrinking, it raises the transaction tax and dials down buybacks. "
            "If cash is growing, it relaxes fees and spends harder to acquire tokens."
        ),
    )
    
    # Fee capture rate controls
    fee_capture_rate = float(autopilot_cfg.get("fee_capture_rate", 0.12))
    override_financing = bool(autopilot_cfg.get("override_financing", False))
    
    cash_tol_pct = float(autopilot_cfg.get("cash_tolerance_pct", 1.5))
    min_tax_pct = float(autopilot_cfg.get("min_tax_pct", max(0.1, base_tax_pct * 0.6)))
    max_tax_pct = float(autopilot_cfg.get("max_tax_pct", max(3.0, base_tax_pct * 2.8)))
    cash_buffer_pct = float(autopilot_cfg.get("cash_buffer_pct", 40.0))
    surplus_spend_rate_pct = float(autopilot_cfg.get("surplus_spend_rate_pct", 20.0))
    fee_step_pct = float(autopilot_cfg.get("fee_step_pct", 0.35))
    buy_push_strength = float(autopilot_cfg.get("buy_push", 0.6))
    buy_brake_strength = float(autopilot_cfg.get("buy_brake", 0.5))
    oracle_enabled = bool(autopilot_cfg.get("oracle_enabled", False))
    oracle_horizon_steps = int(np.clip(autopilot_cfg.get("oracle_horizon_steps", 6), 1, 48))
    oracle_goal = str(autopilot_cfg.get("oracle_goal", "market_cap"))
    if oracle_goal not in {"market_cap", "token_price"}:
        oracle_goal = "market_cap"
    oracle_weight = float(np.clip(autopilot_cfg.get("oracle_weight", 1.0), 0.1, 5.0))
    
    if growth_auto_enabled:
        st.markdown("**Treasury Cash Inflow**")
        fee_cols = st.columns([2, 1])
        with fee_cols[0]:
            fee_capture_rate = st.slider(
                "Fee capture rate (% of transaction volume)",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                value=float(np.clip(fee_capture_rate * 100.0, 0.0, 50.0)),
                help=(
                    "Percentage of trading volume (in USD) that flows to the treasury. "
                    "This is how the treasury accumulates cash to fund buybacks. "
                    "Example: 12% means treasury gets $12 for every $100 in trades."
                ),
            ) / 100.0
        with fee_cols[1]:
            override_financing = st.checkbox(
                "Override Central Bank",
                value=override_financing,
                help=(
                    "When checked, this fee capture rate replaces the 'Financing skim rate' in Central Bank settings. "
                    "When unchecked, both sources contribute to treasury cash."
                ),
            )
        
        st.markdown("**Fee & Buy Pressure Management**")
        cash_tol_pct = st.slider(
            "Cash bleed tolerance (% of treasury per step)",
            min_value=0.2,
            max_value=5.0,
            step=0.1,
            value=float(np.clip(cash_tol_pct, 0.2, 5.0)),
            help=(
                "The autopilot ignores wiggles smaller than this share of treasury cash. "
                "Example: 1.5% means the vault can lose up to 1.5% of its balance in a step before fees spike."
            ),
        )
        min_tax_pct = st.slider(
            "Floor tax rate (%)",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=float(np.clip(min_tax_pct, 0.0, 10.0)),
            help=(
                "Lowest transaction tax the autopilot will allow when cash is flowing. "
                "Keep it above 0.0 if you always want a trickle refilling the treasury."
            ),
        )
        max_tax_pct = st.slider(
            "Ceiling tax rate (%)",
            min_value=0.5,
            max_value=25.0,
            step=0.5,
            value=float(np.clip(max_tax_pct, 0.5, 25.0)),
            help=(
                "Hard cap on emergency fees when cash is bleeding. "
                "Use higher numbers if you are comfortable taxing trades heavily during stress."
            ),
        )
        if max_tax_pct < min_tax_pct:
            max_tax_pct = min_tax_pct
        fee_step_pct = st.slider(
            "Fee adjustment per step (percentage points)",
            min_value=0.05,
            max_value=2.0,
            step=0.05,
            value=float(np.clip(fee_step_pct, 0.05, 2.0)),
            help=(
                "How quickly the autopilot nudges taxes up/down each step it detects a cash bleed or surplus. "
                "Example: 0.35 = move fees by 0.35 percentage points per step."
            ),
        )
        buffer_cols = st.columns(2)
        with buffer_cols[0]:
            cash_buffer_pct = st.slider(
                "Target cash buffer (%)",
                min_value=0.0,
                max_value=200.0,
                step=1.0,
                value=float(np.clip(cash_buffer_pct, 0.0, 200.0)),
                help=(
                    "How much headroom above the moving baseline the treasury keeps before buybacks kick in. "
                    "Set this low (e.g. 5%) to recycle cash aggressively."
                ),
            )
        with buffer_cols[1]:
            surplus_spend_rate_pct = st.slider(
                "Surplus spend rate (% per step)",
                min_value=1.0,
                max_value=100.0,
                step=1.0,
                value=float(np.clip(surplus_spend_rate_pct, 1.0, 100.0)),
                help=(
                    "Share of the accumulated surplus bucket the autopilot redeploys each step. "
                    "Higher values flush cash faster once a buffer build-up is detected."
                ),
            )
        buy_cols = st.columns(2)
        with buy_cols[0]:
            buy_push_strength = st.slider(
                "Buyback boost when flush",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                value=float(np.clip(buy_push_strength, 0.0, 2.0)),
                help=(
                    "Scales extra buy pressure when cash is abundant. "
                    "Higher values = spend faster to chase growth."
                ),
            )
        with buy_cols[1]:
            buy_brake_strength = st.slider(
                "Buyback brake when bleeding",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                value=float(np.clip(buy_brake_strength, 0.0, 2.0)),
                help=(
                    "How strongly to ease off buys when the treasury is shrinking. "
                    "Set to 0 if you prefer fees to rise but buys to keep firing."
                ),
            )

        st.markdown("**Oracle assist (experimental)**")
        oracle_enabled = st.checkbox(
            "Enable oracle assist",
            value=oracle_enabled,
            help=(
                "When enabled, the autopilot projects the goal metric a few steps ahead and biases fee & buy decisions "
                "toward the best expected outcome."
            ),
        )
        if oracle_enabled:
            oracle_cols = st.columns(3)
            with oracle_cols[0]:
                oracle_horizon_steps = st.slider(
                    "Lookahead horizon (steps)",
                    min_value=1,
                    max_value=24,
                    value=int(np.clip(oracle_horizon_steps, 1, 24)),
                    help="How many simulation steps the oracle uses when projecting outcomes.",
                )
            with oracle_cols[1]:
                goal_options = {"market_cap": "Maximise market cap", "token_price": "Maximise token price"}
                oracle_goal = st.selectbox(
                    "Oracle goal",
                    list(goal_options.keys()),
                    index=list(goal_options.keys()).index(oracle_goal if oracle_goal in goal_options else "market_cap"),
                    format_func=lambda key: goal_options[key],
                )
            with oracle_cols[2]:
                oracle_weight = st.slider(
                    "Bias strength",
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    value=float(np.clip(oracle_weight, 0.1, 5.0)),
                    help="Scaling applied to the oracle bias when nudging fees and buybacks.",
                )
            oracle_accuracy = st.slider(
                "Oracle accuracy",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(np.clip(growth_oracle_accuracy, 0.0, 1.0)),
                help="1.0 = follow the best-scoring tweak exactly, 0.0 = stick with rule-based behaviour.",
            )
        else:
            oracle_weight = float(np.clip(oracle_weight, 0.1, 5.0))
            oracle_accuracy = float(np.clip(growth_oracle_accuracy, 0.0, 1.0))
    else:
        oracle_accuracy = float(np.clip(growth_oracle_accuracy, 0.0, 1.0))

    lever_defaults = AlgorithmSettings().leverage_controls
    current_levers = dict(getattr(algo_cfg, "leverage_controls", lever_defaults))
    st.markdown("**Lever permissions**")
    lever_cols = st.columns(5)
    current_levers["buy"] = lever_cols[0].checkbox("buybacks", value=current_levers.get("buy", True))
    current_levers["sell"] = lever_cols[1].checkbox("treasury sells", value=current_levers.get("sell", True))
    current_levers["mint_adjust"] = lever_cols[2].checkbox("mint adjust", value=current_levers.get("mint_adjust", True))
    current_levers["burn_adjust"] = lever_cols[3].checkbox("burn adjust", value=current_levers.get("burn_adjust", True))
    current_levers["hype_boost"] = lever_cols[4].checkbox("hype boost", value=current_levers.get("hype_boost", True))
    algo_cfg.leverage_controls = current_levers
    max_tax_pct = max(max_tax_pct, min_tax_pct)
    algo_cfg.custom_params["growth_autopilot"] = {
        "enabled": growth_auto_enabled,
        "fee_capture_rate": float(fee_capture_rate),
        "override_financing": bool(override_financing),
        "cash_tolerance_pct": float(cash_tol_pct),
        "min_tax_pct": float(min_tax_pct),
        "max_tax_pct": float(max_tax_pct),
        "fee_step_pct": float(fee_step_pct),
        "buy_push": float(buy_push_strength),
        "buy_brake": float(buy_brake_strength),
        "cash_buffer_pct": float(cash_buffer_pct),
        "surplus_spend_rate_pct": float(surplus_spend_rate_pct),
        "oracle_enabled": bool(oracle_enabled),
        "oracle_horizon_steps": int(oracle_horizon_steps),
        "oracle_goal": str(oracle_goal),
        "target_metric": str(growth_oracle_goal),
        "oracle_weight": float(oracle_weight),
        "oracle_accuracy": float(oracle_accuracy),
    }
    # Persist a clean snapshot so the Simulation tab always runs with the latest selections
    st.session_state["algo_settings_snapshot"] = asdict(algo_cfg)

    st.divider()
    st.markdown("##### Custom objectives (optional)")
    st.caption(
        "Need something exotic? Roll your own objective without touching the clunky table. "
        "Anything you add here stacks on top of the preset blocks above."
    )

    removal_indices: List[int] = []
    if advanced_objectives:
        for idx, objective in enumerate(advanced_objectives):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(
                    f"**{objective.name or 'Custom'}** · metric `{objective.metric}` · comparison `{objective.comparison}` · "
                    f"target `{objective.target_value:.3f}` · tolerance `{objective.tolerance:.3f}` · weight `{objective.weight:.2f}`"
                )
            with cols[1]:
                if st.button("Remove", key=f"remove_custom_obj_{idx}"):
                    removal_indices.append(idx)
        for idx in sorted(removal_indices, reverse=True):
            advanced_objectives.pop(idx)
        if removal_indices:
            st.success("Custom objective removed.")
    else:
        st.info("No extra objectives loaded. Enable one above or add a new custom rule below.", icon="ℹ️")

    with st.form("add_custom_objective_form", clear_on_submit=True):
        st.markdown("Add a fresh custom objective")
        custom_name = st.text_input("Label", help="Shown in charts/logs so future-you remembers what this objective does.")
        custom_metric = st.selectbox(
            "Metric",
            ["peg_deviation", "price", "growth", "volatility", "treasury_nav", "velocity", "custom_index"],
            help=(
                "`peg_deviation` uses % away from NAV, `growth` uses per-step adoption rate, "
                "`velocity` is rate-of-change, and `custom_index` is a blend of peg/vol/velocity."
            ),
        )
        custom_comparison = st.selectbox(
            "Comparison",
            ["track", "at_most", "at_least", "maximize", "minimize"],
            help=(
                "Track keeps the metric glued to target. At most/at least enforce ceilings/floors. "
                "Maximize/minimize tell the bot to push without a hard target."
            ),
        )
        custom_target = st.number_input(
            "Target value",
            value=float(0.0),
            help="Exact units depend on the metric you picked. Example: 0.02 for a 2% peg deviation band, 2000 for price.",
        )
        custom_tolerance = st.number_input(
            "Tolerance / sensitivity",
            value=float(0.05),
            min_value=0.0,
            help="How much wiggle room before the bot reacts. Set small values when you want a hair-trigger.",
        )
        custom_weight = st.number_input(
            "Weight",
            value=float(1.0),
            min_value=0.0,
            help="Relative priority. Use >1.5 to make this custom rule outrank the presets above.",
        )
        custom_horizon = st.number_input(
            "Horizon (years)",
            value=float(0.5),
            min_value=0.1,
            help="How long the bot considers the metric trend before judging success.",
        )
        submitted = st.form_submit_button("Add objective")
        if submitted:
            advanced_objectives.append(
                AlgorithmObjective(
                    name=custom_name or "Custom Objective",
                    metric=custom_metric,
                    target_value=custom_target,
                    tolerance=custom_tolerance,
                    comparison=custom_comparison,
                    weight=custom_weight,
                    horizon_years=custom_horizon,
                    enabled=True,
                )
            )
            st.success("Custom objective added.")

    algo_cfg.objectives = curated_objectives + advanced_objectives or [AlgorithmObjective()]

    st.markdown("#### Crash defense mode")
    st.caption(
        "When a waterfall hits, this playbook diverts treasury firepower into buys, gas subsidies, and harder circuit locks. "
        "It only spends what you authorise and cools off once the panic subsides."
    )
    crash_cfg = algo_cfg.crash_defense if hasattr(algo_cfg, "crash_defense") else CrashDefenseSettings()
    crash_top_cols = st.columns(3)
    crash_cfg.enabled = crash_top_cols[0].checkbox(
        "Enable crash defense",
        value=crash_cfg.enabled,
        help="Arms the emergency stabiliser. Leave it off if you want the algorithm to ride crashes unassisted.",
    )
    crash_cfg.drop_threshold_pct = crash_top_cols[1].slider(
        "Crash trigger drop (%)",
        min_value=5.0,
        max_value=50.0,
        step=0.5,
        value=float(np.clip(crash_cfg.drop_threshold_pct, 5.0, 50.0)),
        help="How big a peak-to-now drawdown (over the detection window) sets off the defense mode.",
    )
    crash_cfg.detection_window = crash_top_cols[2].number_input(
        "Detection window (steps)",
        min_value=2,
        max_value=48,
        value=int(np.clip(crash_cfg.detection_window, 2, 48)),
        step=1,
        help="How many recent steps are scanned when looking for a sudden drop.",
    )
    crash_mid_cols = st.columns(3)
    crash_cfg.resource_commitment_pct = crash_mid_cols[0].slider(
        "Resource commitment (%)",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        value=float(np.clip(crash_cfg.resource_commitment_pct, 0.0, 100.0)),
        help="Maximum share of treasury cash the defense can burn through during one crash event.",
    )
    crash_cfg.aggression_pct = crash_mid_cols[1].slider(
        "Aggression throttle (%)",
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        value=float(np.clip(crash_cfg.aggression_pct, 0.0, 100.0)),
        help="Higher = spends the authorised budget faster; 100% means it dumps everything immediately.",
    )
    crash_cfg.gas_subsidy_share_pct = crash_mid_cols[2].slider(
        "Gas subsidy share (%)",
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        value=float(np.clip(crash_cfg.gas_subsidy_share_pct, 0.0, 100.0)),
        help="Slice of the crash budget reserved for covering user gas fees instead of pure buy pressure.",
    )
    crash_bottom_cols = st.columns(4)
    crash_cfg.circuit_lock_steps = crash_bottom_cols[0].number_input(
        "Circuit lock steps",
        min_value=0,
        max_value=48,
        value=int(np.clip(crash_cfg.circuit_lock_steps, 0, 48)),
        step=1,
        help="Extra breaker cooldown steps enforced whenever the defense fires.",
    )
    crash_cfg.stabilize_steps = crash_bottom_cols[1].number_input(
        "Defense duration (steps)",
        min_value=1,
        max_value=36,
        value=int(np.clip(crash_cfg.stabilize_steps, 1, 36)),
        step=1,
        help="How many consecutive steps the crash playbook stays active once triggered.",
    )
    crash_cfg.cooldown_steps = crash_bottom_cols[2].number_input(
        "Re-arm cooldown (steps)",
        min_value=1,
        max_value=72,
        value=int(np.clip(crash_cfg.cooldown_steps, 1, 72)),
        step=1,
        help="Minimum wait before the defense can trigger again after standing down.",
    )
    crash_cfg.gas_efficiency = crash_bottom_cols[3].slider(
        "Gas-to-flow efficiency",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=float(np.clip(crash_cfg.gas_efficiency, 0.0, 1.0)),
        help="Rough conversion from USD gas subsidies into token buying pressure (1.0 = every dollar moves a full dollar of flow).",
    )
    algo_cfg.crash_defense = crash_cfg

    st.markdown("#### Modules")
    modules = algo_cfg.modules
    module_cols = st.columns(3)
    with module_cols[0]:
        modules.circuit_breaker = st.checkbox(
            "Circuit breaker",
            value=modules.circuit_breaker,
            help=(
                "Momentarily slows or pauses the bot once the peg blows past a set threshold. "
                "Useful to stop feedback loops when volatility spikes."
            ),
        )
        modules.circuit_threshold = st.number_input(
            "Breaker threshold",
            value=float(modules.circuit_threshold),
            min_value=0.0,
            step=0.01,
            help=(
                "Deviation level (expressed as a fraction, so 0.07 = 7%) that trips the breaker. "
                "Hit this and the bot instantly shifts into damage-control mode."
            ),
        )
        modules.circuit_cooldown_steps = st.number_input(
            "Breaker cooldown (steps)",
            value=int(modules.circuit_cooldown_steps),
            min_value=0,
            step=1,
            help=(
                "How many simulation steps the breaker stays active once triggered. "
                "Set 4 when running 12 steps/year ≈ 4 months of calmer behaviour."
            ),
        )
        modules.liquidity_support = st.checkbox(
            "Liquidity padding",
            value=modules.liquidity_support,
            help=(
                "Adds synthetic depth to order books, which softens price impact and makes it harder for attackers to move the peg. "
                "Great when organic liquidity is thin."
            ),
        )
    with module_cols[1]:
        modules.adaptive_mint = st.checkbox(
            "Adaptive mint",
            value=modules.adaptive_mint,
            help=(
                "Lets the bot crank up emissions when price runs hot or supply is too tight. "
                "Pairs nicely with growth objectives to reward new users as hype builds."
            ),
        )
        modules.adaptive_burn = st.checkbox(
            "Adaptive burn",
            value=modules.adaptive_burn,
            help=(
                "Authorises emergency burns when price sagging below NAV needs a lift. "
                "This is your \"delete float\" panic button—use it when you want upward pressure fast."
            ),
        )
        modules.trend_follow = st.checkbox(
            "Trend follow",
            value=modules.trend_follow,
            help=(
                "Adds a light momentum bias so interventions lean into strong trends instead of instantly fighting them. "
                "Great for growth runs where you want to ride the wave before reinforcing the peg."
            ),
        )
        modules.velocity_targets = st.checkbox(
            "Velocity targets",
            value=modules.velocity_targets,
            help=(
                "Keeps price velocity hovering near zero by leaning against persistent drift. "
                "Think of it as cruise control for the rate-of-change in price."
            ),
        )
    with module_cols[2]:
        modules.alpha_stabilizer = st.checkbox(
            "Alpha stabilizer",
            value=modules.alpha_stabilizer,
            help=(
                "Adds extra mean-reversion muscle whenever the algorithm is already stepping in. "
                "Use when attackers feed on volatility—you punch back harder the moment you react."
            ),
        )
        modules.drawdown_guard = st.checkbox(
            "Drawdown guard",
            value=modules.drawdown_guard,
            help=(
                "Triggers a higher-aggression playbook after large peak-to-trough crashes. "
                "Prevents slow bleed-outs by flooding support right after a dump."
            ),
        )
        modules.drawdown_threshold = st.slider(
            "Drawdown trigger",
            min_value=0.05,
            max_value=0.6,
            value=float(modules.drawdown_threshold),
            step=0.01,
            help=(
                "Percent drop from the recent high (e.g. 0.18 = 18%) that escalates the drawdown guard. "
                "Lower values = bot panics earlier; higher values = waits for deeper crashes."
            ),
        )
        modules.objective_lock_when_met = st.checkbox(
            "Coast when objectives met",
            value=modules.objective_lock_when_met,
            help=(
                "Dial interventions way down once objectives are green across the board. "
                "Useful to avoid overfitting when the market is already where you want it."
            ),
        )
        anchor_options = ["hybrid", "gold", "fundamental", "none"]
        modules.peg_anchor = st.selectbox(
            "Primary anchor",
            anchor_options,
            index=anchor_options.index(modules.peg_anchor if modules.peg_anchor in anchor_options else "hybrid"),
            help=(
                "Pick the main price truth the bot respects: hybrid (blend of gold + fundamentals), gold-only, "
                "pure fundamentals, or none for free-market experimentation."
            ),
        )

    price_guard_cols = st.columns(2)
    with price_guard_cols[0]:
        floor_price_value = st.number_input(
            "Soft floor price",
            value=float(modules.floor_price or 0.0),
            min_value=0.0,
            step=10.0,
            help=(
                "Optional USD floor where the algorithm opens the cheque book. "
                "Set 0 to disable. Example: 900 means it catches knives below $900."
            ),
        )
        modules.floor_price = floor_price_value if floor_price_value > 0 else None
    with price_guard_cols[1]:
        ceiling_price_value = st.number_input(
            "Soft ceiling price",
            value=float(modules.ceiling_price or 0.0),
            min_value=0.0,
            step=10.0,
            help=(
                "Optional USD ceiling where the bot shifts into sell-the-rally mode. "
                "0 disables it. Example: 2_500 keeps price from ripping past $2.5k without resistance."
            ),
        )
        modules.ceiling_price = ceiling_price_value if ceiling_price_value > 0 else None

    algo_cfg.modules = modules

    st.subheader("Legacy Treasury Modules")
    st.markdown(
        "Keep these around for back-compat or fine-grained tuning. The algorithm above can run with the treasury disabled if you want a pure-strategy run."
    )
    policy.enabled = st.checkbox(
        "Activate peg-defense treasury",
        value=policy.enabled,
        help=(
            "Legacy ruleset that mints, burns, and applies fees using the older logic. "
            "Turn it off for a pure algorithmic run; switch it on to layer traditional levers on top."
        ),
    )

    module_cols = st.columns(3)
    with module_cols[0]:
        policy.module_fee_incentives = st.checkbox(
            "Fee incentives",
            value=policy.module_fee_incentives,
            help=(
                "Rebates fees for trades that push price toward NAV and adds surcharges for trades that drift away. "
                "Works best when you have steady organic volume."
            ),
            disabled=not policy.enabled,
        )
        policy.module_savings = st.checkbox(
            "Savings boost",
            value=policy.module_savings,
            help=(
                "Offers a juiced yield/reserve rate when the token is cheap, convincing loyal holders to stay put instead of panic-selling."
            ),
            disabled=not policy.enabled,
        )
    with module_cols[1]:
        policy.module_liquidity_support = st.checkbox(
            "Virtual liquidity support",
            value=policy.module_liquidity_support,
            help=(
                "Temporarily adds depth on both sides of the book, mimicking market makers stepping in. "
                "Useful when spreads blow out during shocks."
            ),
            disabled=not policy.enabled,
        )
        policy.module_policy_arbitrage = st.checkbox(
            "Policy arbitrage flow",
            value=policy.module_policy_arbitrage,
            help=(
                "Lets the treasury lean with or against price direction (like a sovereign wealth fund) when deviations appear. "
                "Great for nudging price back without outright minting/burning."
            ),
            disabled=not policy.enabled,
        )
    with module_cols[2]:
        policy.module_omo = st.checkbox(
            "Open market ops",
            value=policy.module_omo,
            help=(
                "Classical central-bank open-market operations—buy or sell inventory as soon as the peg slips. "
                "Aggressive but reliable."
            ),
            disabled=not policy.enabled,
        )
        policy.module_gas_subsidy = st.checkbox(
            "Gas subsidies",
            value=policy.module_gas_subsidy,
            help=(
                "Pays users back a slice of transaction costs while defending the peg. "
                "Handy on high-fee chains where arbitrageurs hesitate to step in."
            ),
            disabled=not policy.enabled,
        )
        policy.module_mint_control = st.checkbox(
            "Mint governor",
            value=policy.module_mint_control,
            help=(
                "Gives the policy engine veto power over baseline emission schedules—slowing or accelerating minting as conditions change."
            ),
            disabled=not policy.enabled,
        )

    policy.module_circuit_breaker = st.checkbox(
        "Circuit breaker",
        value=policy.module_circuit_breaker,
        help=(
            "Throttle trading and nudge flows away from the peg when it blows out despite policy tools. "
            "Think of it as the old-school emergency stop."
        ),
        disabled=not policy.enabled,
    )

    col1, col2 = st.columns(2)
    with col1:
        policy.nav_band_soft = st.number_input(
            "Target band (soft)",
            value=policy.nav_band_soft,
            step=0.002,
            format="%.4f",
            help=(
                "Deviation (fractional) where mild incentives kick in. Example: 0.015 = 1.5% gap before rebates/discounts appear."
            ),
        )
        policy.nav_band_hard = st.number_input(
            "Target band (hard)",
            value=policy.nav_band_hard,
            step=0.005,
            format="%.4f",
            help=(
                "Point where the heavy artillery comes out—bigger rebates, steeper penalties, stronger OMO. "
                "Keep this higher than the soft band so responses escalate smoothly."
            ),
        )
        policy.fee_rebate_strength = st.number_input(
            "Fee rebate strength",
            value=policy.fee_rebate_strength,
            step=0.05,
            help=(
                "Multiplier for rewarding trades that collapse the peg gap. "
                "Example: 0.4 = 40% fee rebate when buys/sells improve deviation."
            ),
        )
        policy.fee_penalty_strength = st.number_input(
            "Fee penalty strength",
            value=policy.fee_penalty_strength,
            step=0.05,
            help=(
                "How punishing trades become when they push price further away. "
                "Higher numbers deter reckless dumping during fragile periods."
            ),
        )
        policy.savings_strength = st.number_input(
            "Savings boost (discount side)",
            value=policy.savings_strength,
            step=0.05,
            help=(
                "Temporary APR bonus offered when price is below NAV. "
                "Translate 0.25 as 25% extra yield layered on top of your baseline staking rewards."
            ),
        )
    with col2:
        policy.omo_strength = st.number_input(
            "OMO strength",
            value=policy.omo_strength,
            step=0.05,
            help=(
                "Sensitivity of open-market operations. 0.35 means the treasury spends about 35% of the peg gap per step to close it."
            ),
        )
        policy.max_omo_fraction = st.number_input(
            "OMO clip per step",
            value=policy.max_omo_fraction,
            step=0.05,
            help=(
                "Cap on how much of the treasury's inventory is allowed to move each step. "
                "Example: 0.25 = at most 25% of cash/tokens deployed per period, preventing sudden depletion."
            ),
        )
        policy.breaker_threshold = st.number_input(
            "Breaker trigger",
            value=policy.breaker_threshold,
            step=0.01,
            help=(
                "Deviation (fractional, e.g. 0.05 = 5%) that activates the policy circuit breaker. "
                "Keep this aligned with the module breaker for consistent behaviour."
            ),
        )
        policy.breaker_flow_shock = st.number_input(
            "Breaker severity",
            value=policy.breaker_flow_shock,
            step=0.05,
            help=(
                "How heavy the flow drag becomes when the breaker is on. "
                "0.65 = flows are throttled by 65%, blunting both organic and attacker activity."
            ),
        )
        policy.gas_subsidy_pool = st.number_input(
            "Gas subsidy pool (USD)",
            value=policy.gas_subsidy_pool,
            step=2_500.0,
            help=(
                "Total USD earmarked for paying back network fees while defending the peg. "
                "Increase this on higher-fee chains or longer campaigns."
            ),
        )
        policy.gas_subsidy_rate = st.number_input(
            "Gas subsidy rate",
            value=policy.gas_subsidy_rate,
            step=0.005,
            format="%.4f",
            help=(
                "Fraction of the pool spent each step while the peg sits in the hard band. "
                "Example: 0.02 = burn 2% of the pool every period under stress."
            ),
        )

    with st.expander("Advanced defense levers", expanded=False):
        st.caption("Fine-tune how the peg-defense bot intervenes.")
        policy.reversion_bonus = st.number_input(
            "Additional reversion boost",
            value=policy.reversion_bonus,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="Extra portion of the gap closed when the treasury has teeth.",
        )
        policy.arb_flow_bonus = st.number_input(
            "Policy arbitrage flow strength",
            value=policy.arb_flow_bonus,
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            help="How aggressively policy rails push token flow against deviations.",
        )
        policy.liquidity_support_strength = st.number_input(
            "Liquidity support multiplier",
            value=policy.liquidity_support_strength,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="Extra virtual depth injected when the peg is wobbling.",
        )
        st.markdown("**Activation & ramp**")
        act1, act2 = st.columns(2)
        with act1:
            policy.activation_step = st.number_input(
                "Minimum step before activation",
                value=policy.activation_step,
                min_value=0,
                step=1,
                help="Delay launching the central bank until this step.",
            )
            policy.activation_price = st.number_input(
                "Activation price floor",
                value=policy.activation_price,
                min_value=0.0,
                step=50.0,
                help="Central bank stays dormant until price clears this level (0 = ignore).",
            )
            policy.activation_confidence = st.number_input(
                "Activation confidence floor",
                value=policy.activation_confidence,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Central bank waits until confidence exceeds this threshold.",
            )
        with act2:
            policy.ramp_up_steps = st.number_input(
                "Ramp-up steps",
                value=policy.ramp_up_steps,
                min_value=0,
                step=1,
                help="Number of steps to scale intervention strength from 0 → 100%.",
            )
            policy.bootstrap_tokens = st.number_input(
                "Bootstrap tokens",
                value=policy.bootstrap_tokens,
                step=5_000.0,
                help="One-time token injection when the central bank first activates.",
            )
            policy.bootstrap_cash = st.number_input(
                "Bootstrap cash (USD)",
                value=policy.bootstrap_cash,
                step=5_000.0,
                help="One-time cash top-up when the central bank first activates.",
            )
            policy.mint_ramp_strength = st.number_input(
                "Mint control ramp strength",
                value=policy.mint_ramp_strength,
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                help="How aggressively the central bank nudges issuance up or down.",
                disabled=not (policy.enabled and policy.module_mint_control),
            )
            policy.mint_support_floor = st.number_input(
                "Mint support floor",
                value=policy.mint_support_floor,
                min_value=0.0,
                step=0.05,
                help="Price where minting begins to receive support boosts.",
                disabled=not (policy.enabled and policy.module_mint_control),
            )
            policy.mint_support_ceiling = st.number_input(
                "Mint support ceiling",
                value=policy.mint_support_ceiling,
                min_value=0.0,
                step=0.05,
                help="Price where mint throttling starts to kick in.",
                disabled=not (policy.enabled and policy.module_mint_control),
            )
        st.markdown("**Financing & failure modes**")
        fin1, fin2 = st.columns(2)
        with fin1:
            policy.financing_contrib_rate = st.number_input(
                "Financing skim rate",
                value=policy.financing_contrib_rate,
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                help="Fraction of organic flow (in USD) diverted to refill the treasury.",
            )
            policy.financing_pre_stage = st.checkbox(
                "Collect financing pre-activation",
                value=policy.financing_pre_stage,
                help="Skim contributions even before the central bank is live.",
            )
        with fin2:
            policy.bankruptcy_liquidity_hit = st.number_input(
                "Bankruptcy liquidity hit",
                value=policy.bankruptcy_liquidity_hit,
                min_value=0.0,
                max_value=0.95,
                step=0.05,
                help="Depth reduction when the treasury blows up.",
            )
            policy.bankruptcy_confidence_hit = st.number_input(
                "Bankruptcy confidence hit",
                value=policy.bankruptcy_confidence_hit,
                min_value=0.0,
                max_value=0.9,
                step=0.05,
                help="Immediate drop in confidence once the bank is insolvent.",
            )
            policy.bankruptcy_selloff = st.number_input(
                "Bankruptcy fire-sale fraction",
                value=policy.bankruptcy_selloff,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Fraction of depth that panic dumps when the bank fails.",
            )

    return sim, policy


def main() -> None:
    st.set_page_config(page_title="Crypto Dynamics Simulator", layout="wide")
    st.title("Crypto Dynamics Simulator")
    st.caption("Simulate crypto markets, strategies, and adversaries. Design, stress-test, iterate.")
    render_sidebar_guide()
    st.markdown(
        "Configure the world, defenses, and attackers on the dedicated tabs. When you're ready, run the simulation to refresh the charts and tables below."
    )

    if "sim_config" not in st.session_state:
        st.session_state["sim_config"] = SimulationConfig()
    if "policy_config" not in st.session_state:
        st.session_state["policy_config"] = PolicySettings()

    sim_config: SimulationConfig = st.session_state["sim_config"]
    policy_config: PolicySettings = st.session_state["policy_config"]

    sim_defaults = SimulationConfig()
    for field_name in sim_defaults.__dataclass_fields__:
        if not hasattr(sim_config, field_name):
            setattr(sim_config, field_name, getattr(sim_defaults, field_name))

    policy_defaults = PolicySettings()
    for field_name in policy_defaults.__dataclass_fields__:
        if not hasattr(policy_config, field_name):
            setattr(policy_config, field_name, getattr(policy_defaults, field_name))

    if "manual_attacks" not in st.session_state:
        st.session_state["manual_attacks"] = []
    if "attacks_enabled" not in st.session_state:
        st.session_state["attacks_enabled"] = False
    if "random_attack_mode" not in st.session_state:
        st.session_state["random_attack_mode"] = policy_config.random_attack_mode
    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None
    if "last_run_label" not in st.session_state:
        st.session_state["last_run_label"] = ""
    if "attacker_settings" not in st.session_state:
        st.session_state["attacker_settings"] = AttackerSettings()

    manual_attacks: List[Dict[str, Any]] = st.session_state["manual_attacks"]
    attacker_settings: AttackerSettings = st.session_state["attacker_settings"]

    tabs = st.tabs(["Simulation", "Base Market Conditions", "Algorithm Forge", "Attacker Lab"])

    with tabs[1]:
        sim_config = render_market_conditions_tab(sim_config)
        st.session_state["sim_config"] = sim_config

    with tabs[2]:
        sim_config, policy_config = render_algorithm_tab(sim_config, policy_config)
        st.session_state["sim_config"] = sim_config
        st.session_state["policy_config"] = policy_config

    with tabs[3]:
        total_steps = sim_config.years * sim_config.steps_per_year
        enabled, random_mode, manual_attacks, attacker_settings = render_attacker_tab(
            st.session_state["attacks_enabled"],
            manual_attacks,
            total_steps,
            st.session_state["random_attack_mode"],
            attacker_settings,
            st.session_state["last_output"],
        )
        st.session_state["attacks_enabled"] = enabled
        st.session_state["random_attack_mode"] = random_mode
        st.session_state["manual_attacks"] = manual_attacks
        st.session_state["attacker_settings"] = attacker_settings
        policy_config.random_attack_mode = random_mode

    with tabs[0]:
        st.subheader("Simulation")
        st.markdown(
            "Adjust parameters on the other tabs, then run the scenario here to refresh every chart and table."
        )
        st.divider()

        st.markdown("### Run Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            random_seed = st.number_input("Random seed", value=sim_config.random_seed, step=1)
        with col2:
            initial_price = st.number_input(
                "Initial token price",
                value=sim_config.initial_token_price,
                step=0.05,
                format="%.4f",
                help="Launch price before the market takes over.",
            )
        with col3:
            initial_free_float = st.number_input(
                "Initial free float",
                value=sim_config.initial_free_float,
                step=1_000_000.0,
                help="Quick override for tradable supply at launch.",
            )

        if "algo_settings_snapshot" in st.session_state:
            sim_config.algorithm_settings = _coerce_algorithm_settings(st.session_state["algo_settings_snapshot"])
        else:
            st.session_state["algo_settings_snapshot"] = asdict(sim_config.algorithm_settings)

        run_btn = st.button("Run simulation", type="primary")

        if run_btn:
            sim_config.random_seed = int(random_seed)
            sim_config.initial_token_price = float(initial_price)
            sim_config.initial_free_float = float(initial_free_float)
            sim_config.supply_plan.initial_circulating = float(initial_free_float)

            policy_config.random_attack_mode = st.session_state["random_attack_mode"]

            output = run_simulation(
                sim_config,
                policy_config,
                st.session_state["manual_attacks"],
                st.session_state["attacks_enabled"],
                st.session_state["attacker_settings"],
            )
            st.session_state["last_output"] = output
            run_time = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ")
            attack_label = "on" if st.session_state["attacks_enabled"] else "off"
            st.session_state["last_run_label"] = (
                f"Last run {run_time} | Seed {sim_config.random_seed} | Attacks {attack_label}"
            )

            st.success("Simulation complete. Charts updated below.")

        if st.session_state["last_output"] is not None:
            if st.session_state["last_run_label"]:
                st.caption(st.session_state["last_run_label"])
            render_simulation_tab(st.session_state["last_output"])
        else:
            st.write("Configure your scenario, then run the simulation to populate the visualisations.")

    with st.expander("Current configuration snapshot", expanded=False):
        st.json(
            {
                "simulation": asdict(sim_config),
                "policy": asdict(policy_config),
                "attacker": {
                    "enabled": st.session_state["attacks_enabled"],
                    "random_mode": st.session_state["random_attack_mode"],
                    "manual_events": st.session_state["manual_attacks"],
                    "auto": asdict(st.session_state["attacker_settings"]),
                },
            }
        )

def render_attacker_tab(
    enabled: bool,
    manual_attacks: List[Dict[str, Any]],
    total_steps: int,
    random_mode: str,
    attacker_settings: AttackerSettings,
    last_output: Optional[SimulationOutput],
) -> Tuple[bool, str, List[Dict[str, Any]], AttackerSettings]:
    st.subheader("⚔️ Attacker Lab")
    st.markdown(
        "Test your defenses against profit-seeking traders who try to break your peg and make money doing it."
    )
    st.info(
        "Tip: Start with attackers off to validate your baseline. "
        "Add defenses in Algorithm Forge, then turn on light background attacks to stress test."
    )
    st.markdown("---")
    
    enabled = st.checkbox(
        "Enable attacker activity",
        value=enabled,
        help="Disable to freeze all attackers (manual + AI) for a clean baseline run.",
    )
    noise_options = ["off", "light", "medium", "heavy"]
    random_mode = st.selectbox(
        "Background attack noise",
        noise_options,
        index=noise_options.index(random_mode if enabled else "off"),
        disabled=not enabled,
        help="Blend in opportunistic bots in the background. \"Light\" sprinkles a few pokes per year, \"Heavy\" keeps constant pressure.",
    )

    st.markdown("### Strategic Objective")
    objective_labels = {
        "maximize_pnl": "Maximize PnL (classic hedge fund)",
        "maximize_cash": "End campaign rich in cash",
        "maximize_tokens": "Corner supply (token hoarder)",
        "destabilize": "Destabilize market (peg breaker)",
        "custom": "Custom behaviour (use directives)",
    }
    objective_keys = list(objective_labels.keys())
    current_obj = attacker_settings.objective if attacker_settings.objective in objective_labels else "maximize_pnl"
    objective_index = objective_keys.index(current_obj)

    obj_cols = st.columns(4)
    with obj_cols[0]:
        attacker_settings.auto_enabled = st.checkbox(
            "Enable AI desks",
            value=attacker_settings.auto_enabled,
            disabled=not enabled,
            help="Turn on the autonomous attacker brain. When off, only scripted events fire.",
        )
    with obj_cols[1]:
        attacker_settings.time_awareness = st.checkbox(
            "Time-aware",
            value=attacker_settings.time_awareness,
            disabled=not enabled,
            help="If enabled the attacker cares about how much simulation time is left and will wind down or go berserk near the finale.",
        )
    with obj_cols[2]:
        attacker_settings.horizon_years = st.slider(
            "Campaign length (years)",
            min_value=0.25,
            max_value=10.0,
            value=float(attacker_settings.horizon_years),
            step=0.25,
            disabled=not enabled,
            help="How far ahead the attacker plans. Shorter horizons force faster rotations; long horizons favour slow accumulation.",
        )
    with obj_cols[3]:
        attacker_settings.goal_band = st.slider(
            "Price band tolerance",
            min_value=0.02,
            max_value=0.6,
            value=float(attacker_settings.goal_band),
            step=0.01,
            disabled=not enabled,
            help="How wide the comfort zone is around the target price before the attacker switches phases.",
        )

    attacker_settings.objective = st.selectbox(
        "Objective",
        objective_keys,
        index=objective_index,
        format_func=lambda key: objective_labels.get(key, key),
        disabled=not enabled,
        help="Pick the macro goal for the AI syndicate. \"Custom\" leans fully on your directives below.",
    )

    pricing_cols = st.columns(3)
    with pricing_cols[0]:
        attacker_settings.target_price = st.number_input(
            "Target price (USD)",
            value=float(attacker_settings.target_price),
            min_value=0.0,
            step=10.0,
            disabled=not enabled,
            help="Reference level the attacker considers \"fair\". Leave at 0 to auto-derive from fundamentals.",
        )
    with pricing_cols[1]:
        attacker_settings.trigger_price = st.number_input(
            "Trigger price (USD)",
            value=float(attacker_settings.trigger_price),
            min_value=0.0,
            step=10.0,
            disabled=not enabled,
            help="When price crosses this line the attacker arms its \"go time\" routine (pump or dump depending on goal).",
        )
    with pricing_cols[2]:
        attacker_settings.pump_bias = st.slider(
            "Pump bias",
            min_value=0.0,
            max_value=1.0,
            value=float(attacker_settings.pump_bias),
            step=0.05,
            disabled=not enabled,
            help="0 = mostly dumps, 1 = mostly pumps. 0.5 lets the engine decide based on context.",
        )

    attacker_settings.mode_label = st.text_input(
        "Codename",
        value=attacker_settings.mode_label,
        disabled=not enabled,
        help="Optional label shown in charts and trade logs for the aggregated AI desk.",
    )
    attacker_settings.objective_note = st.text_area(
        "Operator notes",
        value=attacker_settings.objective_note,
        height=80,
        disabled=not enabled,
        help="Leave yourself breadcrumbs for why this configuration exists. Stored with the simulation output.",
    )

    st.markdown("### Phase & Risk Tuning")
    phase_cols = st.columns(4)
    with phase_cols[0]:
        attacker_settings.hoard_cash_ratio = st.slider(
            "Cash hoarding ratio",
            0.0,
            0.9,
            float(attacker_settings.hoard_cash_ratio),
            0.05,
            disabled=not enabled,
            help="Target fraction of total deployable value the attacker parks in cash during hoarding phases.",
        )
        attacker_settings.allow_flash_crash = st.checkbox(
            "Allow flash crash tactics",
            value=attacker_settings.allow_flash_crash,
            disabled=not enabled,
            help="When checked the attacker is willing to nuke liquidity in exchange for chaos during final pushes.",
        )
    with phase_cols[1]:
        attacker_settings.hoard_token_ratio = st.slider(
            "Token hoarding ratio",
            0.0,
            0.95,
            float(attacker_settings.hoard_token_ratio),
            0.05,
            disabled=not enabled,
            help="Ceiling fraction of its maximum position the attacker aims to hold when accumulating.",
        )
        attacker_settings.shared_brain_noise = st.slider(
            "Shared brain noise",
            0.0,
            1.0,
            float(attacker_settings.shared_brain_noise),
            0.05,
            disabled=not enabled,
            help="Inject correlated randomness so desks sometimes do the same dumb thing together.",
        )
    with phase_cols[2]:
        attacker_settings.final_push_aggression = st.slider(
            "Final push aggression",
            0.5,
            4.0,
            float(attacker_settings.final_push_aggression),
            0.1,
            disabled=not enabled,
            help="Multiplier applied when the timer is almost out. Bigger values = nastier endgame swings.",
        )
        attacker_settings.retreat_drawdown = st.slider(
            "Retreat drawdown",
            0.05,
            0.6,
            float(attacker_settings.retreat_drawdown),
            0.01,
            disabled=not enabled,
            help="When cumulative losses exceed this fraction the attacker flips into risk-off mode.",
        )
    with phase_cols[3]:
        attacker_settings.cycle_bias = st.slider(
            "Cycle bias",
            0.0,
            1.0,
            float(attacker_settings.cycle_bias),
            0.05,
            disabled=not enabled,
            help="0 = favour dump cycles, 1 = favour pump cycles. Useful for choreographing classic pump → nuke routines.",
        )
        attacker_settings.max_inventory_multiple = st.slider(
            "Max inventory multiple",
            1.0,
            12.0,
            float(attacker_settings.max_inventory_multiple),
            0.5,
            disabled=not enabled,
            help="Upper bound on how many times the base capital the attacker is allowed to hold in tokens.",
        )

    st.markdown("### Execution Envelope")
    exec_cols = st.columns(3)
    with exec_cols[0]:
        attacker_settings.capital = st.number_input(
            "Initial capital (USD)",
            value=float(attacker_settings.capital),
            min_value=0.0,
            step=100_000.0,
            disabled=not enabled,
        )
        attacker_settings.aggression = st.slider(
            "Base aggression",
            0.0,
            2.0,
            float(attacker_settings.aggression),
            0.05,
            disabled=not enabled,
            help="Higher numbers let the attacker lean harder into signals across every phase.",
        )
        attacker_settings.max_step_fraction = st.slider(
            "Max capital per step",
            0.01,
            0.6,
            float(attacker_settings.max_step_fraction),
            0.01,
            disabled=not enabled,
            help="Hard ceiling on how much of its stack the attacker moves in a single model step.",
        )
    with exec_cols[1]:
        attacker_settings.risk_tolerance = st.slider(
            "Risk tolerance",
            0.0,
            2.0,
            float(attacker_settings.risk_tolerance),
            0.05,
            disabled=not enabled,
            help="How quickly the attacker sizes back up after losses.",
        )
        attacker_settings.loss_aversion = st.slider(
            "Loss aversion",
            0.0,
            1.5,
            float(attacker_settings.loss_aversion),
            0.05,
            disabled=not enabled,
            help="Higher values mean even small drawdowns trigger de-risking.",
        )
        attacker_settings.cooloff_steps = st.number_input(
            "Cool-off steps",
            value=int(attacker_settings.cooloff_steps),
            min_value=0,
            step=1,
            disabled=not enabled,
            help="Minimum wait before the same squad can re-engage after a major move.",
        )
    with exec_cols[2]:
        attacker_settings.micro_steps = st.slider(
            "Intrastep micro-iterations",
            min_value=1,
            max_value=20,
            value=int(attacker_settings.micro_steps),
            disabled=not enabled,
            help="Higher values break each model step into more micro decisions, simulating scalpers.",
        )
        attacker_settings.signal_threshold = st.slider(
            "Signal threshold",
            0.0,
            0.35,
            float(attacker_settings.signal_threshold),
            0.01,
            disabled=not enabled,
            help="Minimum perceived edge before a squad acts.",
        )
        attacker_settings.adaptive_sizing = st.checkbox(
            "Adaptive position sizing",
            value=attacker_settings.adaptive_sizing,
            disabled=not enabled,
            help="Let the desk stretch or shrink sizing dynamically based on recent PnL.",
        )
    toggles_cols = st.columns(2)
    with toggles_cols[0]:
        attacker_settings.reinvest_profits = st.checkbox(
            "Reinvest profits",
            value=attacker_settings.reinvest_profits,
            disabled=not enabled,
        )
    with toggles_cols[1]:
        attacker_settings.escalate_on_profit = st.checkbox(
            "Escalate on wins",
            value=attacker_settings.escalate_on_profit,
            disabled=not enabled,
        )

    st.markdown("### Manual Directives")
    st.caption(
        "Hard-code phase changes with friendly cards instead of spreadsheets. Fill only the triggers you need—leave any field blank to ignore it."
    )

    def _format_optional(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, int)):
            return f"{float(value):.4f}".rstrip("0").rstrip(".")
        return str(value)

    def _parse_optional(label: str, raw_value: str) -> Optional[float]:
        cleaned = raw_value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            st.warning(f"Could not parse '{label}' value '{raw_value}'. Ignoring it.", icon="⚠️")
            return None

    current_directives = attacker_settings.manual_directives or []
    directive_scope_options = ["Any squad"] + [profile.name for profile in attacker_settings.profiles]
    known_phases = [
        "accumulate_tokens",
        "accumulate_cash",
        "pump",
        "dump",
        "range_trade",
        "probe",
        "exit",
        "park",
        "scout",
    ]
    updated_directives: List[Dict[str, Any]] = []
    for idx, directive in enumerate(current_directives):
        trigger = directive.get("trigger", {}) if isinstance(directive, dict) else {}
        scope_raw = directive.get("scope", "any")
        scope_label = "Any squad"
        for option in directive_scope_options[1:]:
            if option == scope_raw:
                scope_label = option
        phase_value = directive.get("phase", "")
        header = phase_value or f"Directive {idx + 1}"
        with st.expander(f"Directive #{idx + 1}: {header}", expanded=False):
            cols = st.columns(2)
            with cols[0]:
                scope_choice = st.selectbox(
                    "Applies to",
                    directive_scope_options,
                    index=directive_scope_options.index(scope_label) if scope_label in directive_scope_options else 0,
                    key=f"directive_scope_{idx}",
                    disabled=not enabled,
                    help="Pick a specific attacker squad or leave on Any squad to broadcast the order to everyone.",
                )
                phase_choice = st.selectbox(
                    "Force phase",
                    known_phases,
                    index=known_phases.index(phase_value) if phase_value in known_phases else 0,
                    key=f"directive_phase_{idx}",
                    disabled=not enabled,
                    help="Which behavioural mode to jump into once triggers match. Examples: pump, dump, range_trade.",
                )
                delete_pressed = False
                if st.button(
                    "Remove directive",
                    key=f"remove_directive_{idx}",
                    disabled=not enabled,
                ):
                    delete_pressed = True
            trigger_cols = st.columns(2)
            trigger_inputs: Dict[str, Optional[float]] = {}
            field_meta = [
                ("price_min", "Only once price ≥ (USD)"),
                ("price_max", "Only while price ≤ (USD)"),
                ("step_after", "Only after step #"),
                ("step_before", "Only before step #"),
                ("deviation_min", "Deviation ≥ (fraction, 0.05 = 5%)"),
                ("deviation_max", "Deviation ≤ (fraction)"),
                ("time_remaining_min", "Time remaining ≥ (years)"),
                ("time_remaining_max", "Time remaining ≤ (years)"),
                ("momentum_min", "Momentum ≥ (fraction)"),
            ]
            for column_index, (field, label) in enumerate(field_meta):
                column = trigger_cols[column_index % 2]
                with column:
                    raw_value = _format_optional(trigger.get(field))
                    text_value = st.text_input(
                        label,
                        value=raw_value,
                        key=f"directive_{field}_{idx}",
                        disabled=not enabled,
                        help="Leave blank to ignore this trigger.",
                    )
                    trigger_inputs[field] = None if not enabled else _parse_optional(label, text_value)
                    if not enabled:
                        trigger_inputs[field] = trigger.get(field)
            if delete_pressed:
                st.success("Directive removed.")
                continue
            updated_trigger = {k: v for k, v in trigger_inputs.items() if v is not None}
            updated_directives.append(
                {
                    "scope": scope_choice if scope_choice != "Any squad" else "any",
                    "phase": phase_choice,
                    "trigger": updated_trigger,
                }
            )

    with st.form("add_manual_directive_form", clear_on_submit=True):
        st.markdown("Add a new directive")
        new_scope = st.selectbox(
            "Who should obey?",
            directive_scope_options,
            key="new_directive_scope",
            help="Send to a specific squad or broadcast to everyone.",
        )
        new_phase = st.selectbox(
            "Phase to force",
            known_phases,
            key="new_directive_phase",
            help="Pick the phase the squad should jump to once the conditions below are true.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            price_min_field = st.text_input(
                "Only once price ≥ (USD)",
                key="new_directive_price_min",
                help="Leave blank to ignore. Example: 1500 = trigger only when price is $1,500 or higher.",
            )
            step_after_field = st.text_input(
                "Only after step #",
                key="new_directive_step_after",
                help="Enter a step number (0-based). Leave blank to allow from the start.",
            )
            dev_min_field = st.text_input(
                "Deviation ≥ (fraction)",
                key="new_directive_dev_min",
                help="Example: 0.05 means trigger once we're 5% or more above NAV.",
            )
            time_min_field = st.text_input(
                "Time remaining ≥ (years)",
                key="new_directive_time_min",
                help="Example: 0.5 means only while at least half a year remains in the sim.",
            )
        with col_b:
            price_max_field = st.text_input(
                "Only while price ≤ (USD)",
                key="new_directive_price_max",
                help="Example: 900 keeps the order active only while price stays below $900.",
            )
            step_before_field = st.text_input(
                "Only before step #",
                key="new_directive_step_before",
                help="Stops the directive after this step. Leave blank for no upper bound.",
            )
            dev_max_field = st.text_input(
                "Deviation ≤ (fraction)",
                key="new_directive_dev_max",
                help="Example: 0.02 keeps it active only while within a 2% premium.",
            )
            time_max_field = st.text_input(
                "Time remaining ≤ (years)",
                key="new_directive_time_max",
                help="Example: 0.1 makes it fire only in the final stretch.",
            )
            momentum_min_field = st.text_input(
                "Momentum ≥ (fraction)",
                key="new_directive_momentum_min",
                help="Positive values mean we need upward momentum before switching phase.",
            )
        submitted = st.form_submit_button("Add directive")
        if submitted:
            if not enabled:
                st.warning("Enable attacker activity to add directives.", icon="ℹ️")
            else:
                new_trigger = {}
                for label, raw in [
                    ("price_min", price_min_field),
                    ("price_max", price_max_field),
                    ("step_after", step_after_field),
                    ("step_before", step_before_field),
                    ("deviation_min", dev_min_field),
                    ("deviation_max", dev_max_field),
                    ("time_remaining_min", time_min_field),
                    ("time_remaining_max", time_max_field),
                    ("momentum_min", momentum_min_field),
                ]:
                    parsed = _parse_optional(label, raw)
                    if parsed is not None:
                        new_trigger[label] = parsed
                updated_directives.append(
                    {
                        "scope": new_scope if new_scope != "Any squad" else "any",
                        "phase": new_phase,
                        "trigger": new_trigger,
                    }
                )
                st.success("Directive added.")

    attacker_settings.manual_directives = updated_directives

    st.markdown("### Scripted Event Attacks")
    st.caption(
        "Design specific shocks—massive buys, coordinated dumps, or liquidity rugs—that fire on exact steps no matter what the AI thinks."
    )
    edited_manual_attacks: List[Dict[str, Any]] = []
    for idx, attack in enumerate(manual_attacks or []):
        attack_name = attack.get("name", f"Attack {idx + 1}")
        with st.expander(f"{attack_name} (#{idx + 1})", expanded=False):
            col_left, col_right = st.columns(2)
            with col_left:
                name_value = st.text_input(
                    "Codename",
                    value=attack_name,
                    key=f"attack_name_{idx}",
                    disabled=not enabled,
                    help="Friendly label for charts/logs. Example: 'March Liquidity Rug'.",
                )
                start_step_value = st.number_input(
                    "Start step",
                    value=int(attack.get("start_step", 0)),
                    min_value=0,
                    max_value=max(0, total_steps - 1),
                    step=1,
                    key=f"attack_start_{idx}",
                    disabled=not enabled,
                    help="Simulation step to fire on (0-based). Multiply years × steps/year to gauge total length.",
                )
                duration_value = st.number_input(
                    "Duration (steps)",
                    value=int(attack.get("duration", 1)),
                    min_value=1,
                    max_value=max(1, total_steps),
                    step=1,
                    key=f"attack_duration_{idx}",
                    disabled=not enabled,
                    help="How many consecutive steps the event stays active.",
                )
                side_value = st.selectbox(
                    "Attack style",
                    ["buy", "sell", "mixed"],
                    index=["buy", "sell", "mixed"].index(attack.get("side", "sell") if attack.get("side") in ["buy", "sell", "mixed"] else "sell"),
                    key=f"attack_side_{idx}",
                    disabled=not enabled,
                    help="• buy = coordinated pump, • sell = dump, • mixed = whipsaw plus optional liquidity drain.",
                )
            with col_right:
                magnitude_value = st.number_input(
                    "Size vs liquidity depth",
                    value=float(attack.get("magnitude", 0.05)),
                    min_value=0.0,
                    max_value=1.5,
                    step=0.01,
                    key=f"attack_magnitude_{idx}",
                    disabled=not enabled,
                    help="Fraction of the active liquidity depth to chew through per step. 0.10 ≈ 10% of depth.",
                )
                liquidity_drop_value = st.number_input(
                    "Liquidity rug fraction",
                    value=float(attack.get("liquidity_drop", 0.0)),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key=f"attack_liquidity_{idx}",
                    disabled=not enabled,
                    help="Optional extra rug. 0.35 = pull 35% of depth while the attack runs.",
                )
                notes_value = st.text_area(
                    "Notes / storyboard",
                    value=str(attack.get("notes", "")),
                    key=f"attack_notes_{idx}",
                    height=80,
                    disabled=not enabled,
                    help="Document the narrative so you remember why this attack exists.",
                )
                delete_attack = False
                if st.button(
                    "Remove attack",
                    key=f"remove_attack_{idx}",
                    disabled=not enabled,
                ):
                    delete_attack = True
            if delete_attack:
                st.success("Attack removed.")
                continue
            edited_manual_attacks.append(
                {
                    "name": name_value,
                    "start_step": int(start_step_value),
                    "duration": int(duration_value),
                    "side": side_value,
                    "magnitude": float(magnitude_value),
                    "liquidity_drop": float(liquidity_drop_value),
                    "notes": notes_value,
                }
            )

    with st.form("add_manual_attack_form", clear_on_submit=True):
        st.markdown("Add a scripted attack")
        col_a, col_b = st.columns(2)
        with col_a:
            new_attack_name = st.text_input(
                "Codename",
                key="new_attack_name",
                help="Short name shown in logs, e.g. 'Black Swan Dump'.",
            )
            new_start_step = st.number_input(
                "Start step",
                min_value=0,
                max_value=max(0, total_steps - 1),
                step=1,
                key="new_attack_start",
            )
            new_duration = st.number_input(
                "Duration (steps)",
                min_value=1,
                max_value=max(1, total_steps),
                step=1,
                key="new_attack_duration",
            )
        with col_b:
            new_side = st.selectbox(
                "Attack style",
                ["sell", "buy", "mixed"],
                key="new_attack_side",
                help="Sell = dump, Buy = pump, Mixed = switch sides and optionally yank liquidity.",
            )
            new_magnitude = st.number_input(
                "Size vs liquidity depth",
                min_value=0.0,
                max_value=1.5,
                step=0.01,
                value=0.08,
                key="new_attack_magnitude",
            )
            new_liquidity_drop = st.number_input(
                "Liquidity rug fraction",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=0.0,
                key="new_attack_liquidity_drop",
            )
        new_notes = st.text_area(
            "Notes / storyboard",
            key="new_attack_notes",
            height=80,
            help="Explain the story so you can reproduce it later—e.g. 'Coordinated dump after treasury exhausted'.",
        )
        add_attack = st.form_submit_button("Add scripted attack")
        if add_attack:
            if not enabled:
                st.warning("Enable attacker activity to add scripted events.", icon="ℹ️")
            else:
                edited_manual_attacks.append(
                    {
                        "name": new_attack_name or f"Attack {len(edited_manual_attacks) + 1}",
                        "start_step": int(new_start_step),
                        "duration": int(new_duration),
                        "side": new_side,
                        "magnitude": float(new_magnitude),
                        "liquidity_drop": float(new_liquidity_drop),
                        "notes": new_notes,
                    }
                )
                st.success("Scripted attack added.")

    st.markdown("### Auto-attacker Squads")
    st.caption("Layer multiple desks with their own styles. The AI runtime orchestrates them according to your objective.")
    profile_options = ["pump_and_dump", "momentum", "liquidity_sapper", "arb_sniper"]
    remove_indices: List[int] = []
    for idx, profile in enumerate(attacker_settings.profiles):
        label = f"{profile.name} · {profile.style}"
        with st.expander(label, expanded=False):
            attacker_settings.profiles[idx].name = st.text_input(
                "Codename",
                value=profile.name,
                key=f"profile_name_{idx}",
                disabled=not enabled,
            )
            style_index = profile_options.index(profile.style) if profile.style in profile_options else 0
            attacker_settings.profiles[idx].style = st.selectbox(
                "Strategy archetype",
                options=profile_options,
                index=style_index,
                key=f"profile_style_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].capital = st.number_input(
                "Capital (USD)",
                value=float(profile.capital),
                min_value=0.0,
                step=25_000.0,
                key=f"profile_capital_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].aggression = st.slider(
                "Aggression",
                min_value=0.0,
                max_value=1.8,
                value=float(profile.aggression),
                step=0.05,
                key=f"profile_aggr_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].skill = st.slider(
                "Skill",
                min_value=0.0,
                max_value=1.0,
                value=float(profile.skill),
                step=0.05,
                key=f"profile_skill_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].perception = st.slider(
                "Signal perception",
                min_value=0.0,
                max_value=1.0,
                value=float(profile.perception),
                step=0.05,
                key=f"profile_perception_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].patience = st.slider(
                "Patience",
                min_value=0.0,
                max_value=1.0,
                value=float(profile.patience),
                step=0.05,
                key=f"profile_patience_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].leverage = st.slider(
                "Leverage",
                min_value=0.5,
                max_value=4.0,
                value=float(profile.leverage),
                step=0.1,
                key=f"profile_leverage_{idx}",
                disabled=not enabled,
            )
            attacker_settings.profiles[idx].reinvest = st.checkbox(
                "Reinvest profits",
                value=profile.reinvest,
                key=f"profile_reinvest_{idx}",
                disabled=not enabled,
            )
            if st.button("Remove squad", key=f"profile_remove_{idx}", disabled=not enabled):
                remove_indices.append(idx)

    if st.button("Add new attacker squad", key="add_profile_v2", disabled=not enabled):
        attacker_settings.profiles.append(
            AttackerProfile(
                name=f"Custom Desk {len(attacker_settings.profiles) + 1}",
                style="momentum",
                capital=500_000.0,
                aggression=0.35,
                skill=0.55,
                patience=0.4,
                perception=0.5,
                reinvest=True,
                leverage=1.2,
            )
        )
    for idx in sorted(remove_indices, reverse=True):
        if 0 <= idx < len(attacker_settings.profiles):
            attacker_settings.profiles.pop(idx)

    if last_output and last_output.attacker_state is not None and not last_output.attacker_state.empty:
        st.markdown("### Recent Attacker Telemetry")
        state_df = last_output.attacker_state.copy()
        signal_fig = px.line(
            state_df,
            x="month",
            y=["auto_signal", "attacker_intensity"],
            labels={"value": "Signal / Intensity", "variable": "Series", "month": "Month"},
            template="plotly_white",
        )
        _plotly(signal_fig, full_width=True)

        progress_fig = px.line(
            state_df,
            x="month",
            y="attacker_objective_progress",
            labels={"attacker_objective_progress": "Objective progress", "month": "Month"},
            template="plotly_white",
        )
        _plotly(progress_fig, full_width=True)

        holdings_fig = px.line(
            state_df,
            x="month",
            y=["attacker_cash", "attacker_position_value"],
            labels={"value": "USD", "variable": "Holding", "month": "Month"},
            template="plotly_white",
        )
        _plotly(holdings_fig, full_width=True)

        phase_fig = px.scatter(
            state_df,
            x="month",
            y="attacker_objective_progress",
            color="auto_mode",
            size=np.clip(state_df["attacker_intensity"].abs(), 0.1, None),
            labels={"auto_mode": "Phase", "attacker_objective_progress": "Progress"},
            template="plotly_white",
        )
        phase_fig.update_traces(marker=dict(sizemode="area", sizeref=0.05))
        _plotly(phase_fig, full_width=True)

    return enabled, (random_mode if enabled else "off"), edited_manual_attacks, attacker_settings


if __name__ == "__main__":
    main()
