"""
Gramian Angular Field (GAF) transformation module for Phase 2.

Converts time series to 2D images for CNN processing.
"""

from .gaf_transformer import (
    generate_gasf,
    generate_gadf,
    generate_gaf,
    normalize_timeseries,
    polar_encode
)

from .gaf_dataset import GAFDataset

__all__ = [
    'generate_gasf',
    'generate_gadf',
    'generate_gaf',
    'normalize_timeseries',
    'polar_encode',
    'GAFDataset'
]
