"""Python-facing dedispersion API.

The functions in this module return host-resident NumPy arrays. Device-resident
Ascend result objects remain a C++ API; the Python surface provides synchronous
CPU and Ascend host-input convenience paths.
"""

from ._core import (
    DedispersedResult,
    DedispersedSpectrum,
    dedisperse_multi_dm,
    dedisperse_single_dm,
    dedisperse_spectrum,
    dedisperse_subband,
)

__all__ = [
    "DedispersedResult",
    "DedispersedSpectrum",
    "dedisperse_multi_dm",
    "dedisperse_single_dm",
    "dedisperse_spectrum",
    "dedisperse_subband",
]
