"""Unified D-GNN framework for C. elegans embryogenesis.

Merges NDP-HNN (snapshot-based hypergraph temporal GNN) and DevoTG
(continuous-time dynamic graph with TGN memory) into a single configurable
pipeline with a shared data layer.
"""

from devograph.unified.config import UnifiedConfig
from devograph.unified.graph_builder import UnifiedGraphBuilder
from devograph.unified.model import DevoGNN

__all__ = ["UnifiedConfig", "UnifiedGraphBuilder", "DevoGNN"]
