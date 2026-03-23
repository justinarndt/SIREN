"""
SIREN — Siri Intelligence, Radically Efficient Neural Engine

Block-circulant structured linear algebra for 2000x model compression
targeting Apple Neural Engine SRAM-resident inference.

Copyright (c) 2026 Justin Arndt
"""

__version__ = "0.1.0"

from siren.core.circulant import BlockCirculantLinear, circulant_matvec
from siren.core.quantization import PhaseMagnitudeQuantizer
from siren.models.transformer import SIRENTransformer, SIRENConfig

__all__ = [
    "BlockCirculantLinear",
    "circulant_matvec",
    "PhaseMagnitudeQuantizer",
    "SIRENTransformer",
    "SIRENConfig",
]
