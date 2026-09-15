# GAFFA AscendC Port

This worktree contains the staged CUDA-to-AscendC migration of GAFFA. The target is one portability-first backend for Atlas A2 and Atlas A3 products using the common `dav-2201` AscendC architecture target.

## CANN build

After installing CANN and sourcing its environment:

```bash
cmake -S . -B build -G Ninja -DGAFFA_ENABLE_ASCEND=ON -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The backend uses ACL runtime allocation/copies/streams on the host and AscendC kernels on the device.

## Host-only contract check

A machine without CANN can still validate public contracts, host orchestration, overflow checks, numerical reference behavior, and device-source contracts:

```bash
cmake -S . -B build-host -G Ninja -DGAFFA_ENABLE_ASCEND=OFF -DBUILD_TESTING=ON
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

Host-only mode is a development aid, not a replacement for BiSheng compilation and A2/A3 device tests.

## Current migration status

- [x] Architecture/design and execution plan
- [x] `AscendRuntime` interface and ACL implementation
- [x] Move-only GM memory and typed spans/buffers
- [x] A2/A3 common launch contract (`dav-2201`)
- [x] AscendC vector-add smoke kernel
- [x] Time-series weighted downsampling and uint32-to-float conversion
- [x] Preprocessing: running-median detrend, interpolation/subtraction, finite checks, normalization
- [x] Direct single/multi-DM dedispersion and aligned dynamic-spectrum output
- [x] True two-stage tiled subband dedispersion with residual-delay halo
- [x] FFA prepare/transform/detection/search plus reusable execution plan/Program workspace
- [ ] Python bindings
- [ ] Benchmark migration
- [ ] Final CUDA removal and full fork integration

## Round history

- Round 1: Ascend runtime, GM RAII memory, launch contract, vector-add smoke kernel.
- Round 2: weighted time-series downsampling and uint32-to-float batch conversion.
- Round 3: preprocessing Program/workspace plus exact/fast running median and normalization.
- Round 4: direct single/multi-DM dedispersion and aligned-spectrum APIs/kernels with host-double delay tables.
- Round 5: true two-stage subband dedispersion with tiled intermediate workspace and residual halo.
- Round 6: FFA prepare plus materialized transform using host-precompiled copy/merge levels and A2/A3-common AscendC kernels.
- Round 7: deterministic FFA boxcar detection and end-to-end batch search composition.
- Round 8: grouped execution plan and move-only `AscendFfaProgram` with reusable large GM workspaces.

See `ROUND1_MANIFEST.md` through `ROUND8_MANIFEST.md` for exact verification boundaries. A real CANN/BiSheng + A2/A3 runner is still required before device compilation/execution can be certified.
