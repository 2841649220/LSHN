"""
lshn.core — 核心计算原语 (细胞、突触、可塑性、演化)
"""

from lshn.core.cells import DendriteCompartment, LiquidGatedCell
from lshn.core.synapses import BistableHypergraphSynapse, AxonalDelayModule
from lshn.core.plasticity import (
    ThreeFactorPlasticity,
    PoissonErrorEncoder,
    SynapticScaling,
    IntrinsicExcitabilityPlasticity,
    HomeostaticController,
)
from lshn.core.evolution import PruneGrowthModule

__all__ = [
    "DendriteCompartment",
    "LiquidGatedCell",
    "BistableHypergraphSynapse",
    "AxonalDelayModule",
    "ThreeFactorPlasticity",
    "PoissonErrorEncoder",
    "SynapticScaling",
    "IntrinsicExcitabilityPlasticity",
    "HomeostaticController",
    "PruneGrowthModule",
]
