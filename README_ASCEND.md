# GAFFA AscendC Port

This fork migrates GAFFA's device backend from CUDA to AscendC while retaining
the CPU reference implementations and the non-GPU astronomy/IO layers. The
target is one portability-first backend for **Atlas A2 and Atlas A3** using the
common `dav-2201` AscendC architecture target.

## CANN build

Install a CANN development toolkit, activate the project environment, and source
CANN through the repository helper:

```bash
conda env create -f environment.yml
conda activate gaffa
source env/dev.sh

cmake -S . -B build -G Ninja \
  -DGAFFA_ASCEND_ARCH=dav-2201 \
  -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`env/dev.sh` discovers the CANN setup script and exposes the AscendC CMake
package directory. The build uses `find_package(ASC)`, the `ASC` CMake language,
ACL (`libascendcl.so`) on the host, and `.asc` kernels on the device.

## Migration status

- [x] Architecture/design and execution plan
- [x] `AscendRuntime` and ACL host runtime
- [x] Move-only GM memory and typed spans/buffers
- [x] A2/A3 common build/launch target (`dav-2201`)
- [x] AscendC vector-add smoke path
- [x] Weighted time-series downsampling and uint32-to-float conversion
- [x] Preprocessing Program/workspace: running-median detrend, interpolation,
      finite checks and normalization
- [x] Direct single/multi-DM dedispersion
- [x] Aligned dynamic-spectrum dedispersion
- [x] True two-stage tiled subband dedispersion with residual-delay halo
- [x] FFA prepare, transform, deterministic boxcar detection and batch search
- [x] Reusable grouped `AscendFfaProgram` workspace
- [x] Device-resident dedispersion → preprocessing → FFA search pipeline test
- [x] Python FFA and dedispersion bindings (`backend="ascend"`)
- [x] Ascend dedispersion and reusable-FFA benchmarks
- [x] CANN/BiSheng developer environment and packaging configuration
- [x] Physical removal of legacy CUDA headers, `.cu`/`.cuh` sources, CUDA-only
      tests and mixed CUDA benchmarks
- [x] Integration into the complete upstream fork while retaining CPU/IO,
      candidate, harmonic, folding and filterbank functionality
- [ ] BiSheng compilation on the target CANN release
- [ ] Runtime parity tests on physical Atlas A2
- [ ] Runtime parity tests on physical Atlas A3

The final three unchecked items are hardware/toolchain verification requirements,
not unimplemented migration modules.

## Verification layers

### Source and host-contract tests

The repository contains source-contract tests for each AscendC kernel family.
They assert required A2/A3-safe kernel/launcher symbols and prevent CUDA-only
constructs such as `threadIdx`, `blockIdx`, `__shared__`, warp shuffles, CUB and
CUDA atomics from leaking back into the Ascend backend.

Host-layout GTests cover:

- GM span/buffer overflow contracts
- weighted downsample plans against the CPU numerical reference
- preprocessing workspace/layout validation
- single/multi-DM delay tables against the CPU double-precision definition
- FFA recursive schedule compilation and reusable workspace sizing

During development, the reconstructed host/fake-ACL harness was freshly run with
**22/22 tests passing** after the core migration and multi-device fixes. This is
evidence for host orchestration and numerical contracts; it is not presented as
an A2/A3 runtime result.

### Device parity tests

When an Ascend device is visible, GTest/PyTest device tests stop skipping and
compare Ascend results with the retained CPU references. The end-to-end C++
pipeline test covers:

```text
subband dedispersion (device)
    -> uint32 to float (device)
    -> preprocessing/normalization (device)
    -> FFA transform/detection (device)
    -> compact peaks (host)
```

Peak identity, phase, width, S/N, period and frequency are compared against the
CPU path.

## Numerical design decisions

Dispersion delays remain a host-side double-precision calculation and are
rounded to `int32` delay tables before upload. This preserves the original GAFFA
scientific definition without requiring device FP64 for delay construction.
Integer dedispersion accumulates into `uint32`; float input retains float
accumulation.

The FFA implementation does not translate CUDA warp/shared-memory mechanics
one-for-one. Host code compiles deterministic copy/merge level schedules, AI
Cores operate on independent series/rows, and detection writes deterministic
per-slot phase/SNR records that are compacted on the host rather than through a
single contended global atomic peak buffer.

## Benchmarks

After a release build:

```bash
just bench-dedispersion-ascend
just bench-ffa-ascend
```

The FFA benchmark warms the reusable `AscendFfaProgram` before measuring
steady-state execution so workspace allocation is not included in every
iteration.

## Verification boundary

This repository is now an AscendC source migration, but the current ChatGPT
execution environment does not contain CANN/BiSheng or an Atlas NPU. Therefore
no claim is made that the present branch has already compiled or passed runtime
tests on physical A2/A3 hardware. Production acceptance requires running the
commands above on both target platforms and recording those results.
