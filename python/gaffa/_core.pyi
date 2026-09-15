"""Type stubs for the compiled GAFFA extension.

Most users should import the public modules ``gaffa.io``, ``gaffa.dedispersion``,
``gaffa.ffa`` and ``gaffa.pfold`` instead of importing this module directly.
"""

from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLikeStr: TypeAlias = str | PathLike[str]
Backend: TypeAlias = Literal["cpu", "ascend"]
ChannelOrder: TypeAlias = Literal["frequency_ascending", "preserve_file_order"]
ReverseBackendName: TypeAlias = Literal["auto", "cpu_scalar", "cpu_openmp"]
FilterbankArray: TypeAlias = NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
DedispersedArray: TypeAlias = NDArray[np.uint32] | NDArray[np.float32]
DedispersedSpectrumArray: TypeAlias = (
    NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
)
FoldCubeArray: TypeAlias = NDArray[np.float32]
FoldExposureArray: TypeAlias = NDArray[np.float64]


class ChannelOrderPolicy(Enum):
    FrequencyAscending: ChannelOrderPolicy
    PreserveFileOrder: ChannelOrderPolicy


class ReverseBackend(Enum):
    Auto: ReverseBackend
    CpuScalar: ReverseBackend
    CpuOpenmp: ReverseBackend


class FilterbankHeader:
    header_size: int
    nsamples: int
    telescope_id: int
    machine_id: int
    data_type: int
    barycentric: int
    pulsarcentric: int
    ibeam: int
    nbeams: int
    npuls: int
    nbins: int
    nbits: int
    nifs: int
    nchans: int
    az_start: float
    za_start: float
    src_raj: float
    src_dej: float
    tstart: float
    tsamp: float
    fch1: float
    foff: float
    refdm: float
    period: float
    rawdatafile: str
    source_name: str
    frequency_table: list[float]
    uses_frequency_table: bool
    def __repr__(self) -> str: ...


class Filterbank:
    header: FilterbankHeader
    data: FilterbankArray

    def __init__(
        self,
        path: PathLikeStr,
        *,
        channel_order: ChannelOrder = "frequency_ascending",
        reverse_backend: ReverseBackendName = "auto",
        io_buffer_bytes: int = 67_108_864,
        openmp_min_rows: int = 4096,
    ) -> None: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def dtype(self) -> np.dtype[np.generic]: ...
    @property
    def nbytes(self) -> int: ...
    def __repr__(self) -> str: ...


class DedispersedResult:
    data: DedispersedArray
    backend: str
    dm_low: float
    dm_step: float
    ndm: int
    nsamples: int
    tsamp: float

    def __init__(
        self,
        data: DedispersedArray,
        *,
        tsamp: float,
        dm_low: float = 0.0,
        dm_step: float = 0.0,
        backend: str = "external",
    ) -> None: ...

    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def dtype(self) -> np.dtype[np.generic]: ...
    @property
    def nbytes(self) -> int: ...
    def __repr__(self) -> str: ...


class DedispersedSpectrum:
    data: DedispersedSpectrumArray
    backend: str
    dm: float
    nsamples: int
    nchans: int
    tsamp: float
    chan_begin: int
    chan_end: int

    def __init__(
        self,
        data: DedispersedSpectrumArray,
        *,
        tsamp: float,
        dm: float = 0.0,
        chan_begin: int = 0,
        chan_end: int | None = None,
        backend: str = "external",
    ) -> None: ...

    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def dtype(self) -> np.dtype[np.generic]: ...
    @property
    def nbytes(self) -> int: ...
    def __repr__(self) -> str: ...


class FfaPlan:
    task_count: int
    width_trials: tuple[int, ...]
    def __repr__(self) -> str: ...


class FfaPeak:
    period: float
    frequency: float
    snr: float
    width: int
    duty_cycle: float
    phase: int
    shift: int
    bins: int
    width_index: int
    period_index: int
    def __repr__(self) -> str: ...


class FoldedProfile:
    profile: NDArray[np.float32]
    exposure: NDArray[np.float64]
    phase: NDArray[np.float32]
    nbin: int
    period: float
    tsamp: float
    def __repr__(self) -> str: ...


class FoldResult:
    cube: FoldCubeArray
    exposure: FoldExposureArray
    profile: NDArray[np.float32]
    freq_phase: NDArray[np.float32]
    time_phase: NDArray[np.float32]
    phase: NDArray[np.float32]
    time: NDArray[np.float64]
    nsubint: int
    nchans: int
    nbin: int
    period: float
    tsamp: float
    tsubint: float
    def __repr__(self) -> str: ...


def _make_riptide_ffa_plan(
    *,
    nsamples: int,
    tsamp: float,
    period_min: float,
    period_max: float,
    bins_min: int = 180,
    bins_max: int = 256,
    min_periods: int = 1,
    duty_cycle_max: float = 0.20,
    width_trial_spacing: float = 1.5,
    max_tasks: int = 1_000_000,
) -> FfaPlan: ...


def _ffa_search_cpu(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]: ...


def _ffa_search_ascend_host(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    device_id: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]: ...


def _fold_dedispersed_profile(
    data: DedispersedArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    dm_index: int = 0,
) -> FoldedProfile: ...


def _fold_dedispersed_spectrum(
    data: DedispersedSpectrumArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    tsubint: float,
    output_channels: int,
) -> FoldResult: ...


def dedisperse_spectrum(
    filterbank: Filterbank,
    *,
    dm: float,
    backend: Backend = "cpu",
    device_id: int = 0,
    block_dim: int = 0,
    time_tile_samples: int = 81920,
) -> DedispersedSpectrum: ...


def dedisperse_single_dm(
    filterbank: Filterbank,
    *,
    dm: float,
    backend: Backend = "cpu",
    device_id: int = 0,
    block_dim: int = 0,
    time_tile_samples: int = 81920,
) -> DedispersedResult: ...


def dedisperse_multi_dm(
    filterbank: Filterbank,
    *,
    dm_low: float,
    dm_step: float,
    ndm: int,
    backend: Backend = "cpu",
    device_id: int = 0,
    block_dim: int = 0,
    time_tile_samples: int = 81920,
) -> DedispersedResult: ...


def dedisperse_subband(
    filterbank: Filterbank,
    *,
    dm_low: float,
    dm_step: float,
    ndm: int,
    backend: Backend = "ascend",
    subband_channels: int = 32,
    ndm_per_nominal: int = 32,
    device_id: int = 0,
    block_dim: int = 0,
    time_tile_samples: int = 81920,
) -> DedispersedResult: ...


def vector_add(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]: ...


def ascend_device_count() -> int:
    """Return the number of Ascend devices visible to ACL runtime."""
    ...
