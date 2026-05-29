"""
lshn.core.synapses — 双稳态超图突触与轴突延迟
"""

from lshn.core.synapses.bistable_hypergraph import BistableHypergraphSynapse
from lshn.core.synapses.axonal_delay import AxonalDelayModule

__all__ = ["BistableHypergraphSynapse", "AxonalDelayModule"]
