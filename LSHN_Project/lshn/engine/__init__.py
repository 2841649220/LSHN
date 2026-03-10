"""
lshn.engine — 全局引擎 (时钟同步、自由能、预算控制、主动推理、神经调节器、知识归档)
"""

from lshn.engine.clock_sync import ClockSyncEngine, T_FAST_MS, T_SLOW_MS, T_ULTRA_MS
from lshn.engine.free_energy import FreeEnergyEngine
from lshn.engine.budget_control import SpikeBudgetController
from lshn.engine.active_inference import ActiveInferenceEngine
from lshn.engine.global_modulator import AstrocyteGate, GlobalNeuromodulator
from lshn.engine.knowledge_archiver import KnowledgeArchiver

__all__ = [
    "ClockSyncEngine",
    "T_FAST_MS",
    "T_SLOW_MS",
    "T_ULTRA_MS",
    "FreeEnergyEngine",
    "SpikeBudgetController",
    "ActiveInferenceEngine",
    "AstrocyteGate",
    "GlobalNeuromodulator",
    "KnowledgeArchiver",
]
