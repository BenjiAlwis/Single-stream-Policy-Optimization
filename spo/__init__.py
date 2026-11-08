"""SPO: Single-stream Policy Optimization, generalized to tabular RL.

A faithful reimplementation of the algorithm from "Single-stream Policy
Optimization" (Xu & Ding, NeurIPS 2025 MATH-AI workshop), reduced from the
LLM-RLVR setting to a contextual-bandit / tabular setting where the
"value of a prompt under policy pi" generalizes to "the value of a state
under policy pi".
"""

from .advantage import (
    normalize_advantages,
    raw_advantage,
    sample_states,
    sampling_weights,
)
from .envs import ContextualBandit
from .grpo_trainer import GRPOConfig, GRPOMetrics, GRPOTrainer
from .policy import TabularPolicy
from .spo_trainer import SPOConfig, SPOMetrics, SPOTrainer
from .tracker import BetaState, KLAdaptiveTracker

__all__ = [
    "ContextualBandit",
    "TabularPolicy",
    "KLAdaptiveTracker",
    "BetaState",
    "SPOTrainer",
    "SPOConfig",
    "SPOMetrics",
    "GRPOTrainer",
    "GRPOConfig",
    "GRPOMetrics",
    "sampling_weights",
    "sample_states",
    "raw_advantage",
    "normalize_advantages",
]
