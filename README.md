# gaffa-AscendC

AscendC/C++ and Python library for pulsar data processing and fast-folding
algorithm (FFA) periodicity searches. This fork ports the original GAFFA CUDA
backend to Huawei Ascend while preserving the CPU reference implementations.

The device backend targets **Atlas A2 and Atlas A3** with the common AscendC
architecture target `dav-2201`.

## Requirements

- Python 3.12
- GCC/G++ 13
- CMake >= 3.26 and Ninja
- Conan >= 2
- CANN development toolkit with AscendC/BiSheng and ACL runtime
- Atlas A2 or Atlas A3 NPU for device execution

CANN 8.5 uses `cann/set_env.sh` in the current default installation layout.
`env/dev.sh` also recognizes the older `ascend-toolkit/set_env.sh` layout.

## Environment

```bash
conda env create -f environment.yml
conda activate gaffa
source env/dev.sh
```

If CANN is installed in a non-standard location, point the helper at its setup
script:

```bash
export GAFFA_CANN_SET_ENV=/path/to/cann/set_env.sh
source env/dev.sh
```

## Development

```bash
just install
just test-all
```

To configure/build the C++ project directly:

```bash
source env/dev.sh
cmake -S . -B build/dev -G Ninja \
  -DGAFFA_ASCEND_ARCH=dav-2201 \
  -DBUILD_TESTING=ON
cmake --build build/dev
ctest --test-dir build/dev --output-on-failure
```

## Backend surface

Python APIs use `backend="cpu"` or `backend="ascend"`. Device selection uses
`device_id`; `gaffa.ascend_device_count()` reports visible Ascend devices.

The Ascend backend currently includes:

- GM/runtime RAII and stream management
- weighted time-series downsampling and conversion
- preprocessing and normalization
- direct single/multi-DM dedispersion
- aligned dynamic-spectrum dedispersion
- true two-stage tiled subband dedispersion
- FFA prepare, transform, detection, search, and reusable `AscendFfaProgram`

See [README_ASCEND.md](README_ASCEND.md) for migration details and validation
boundaries.

## Validation note

Host orchestration, numerical reference contracts, overflow checks, and source
contracts have been exercised during the port. Final production acceptance also
requires compiling with the target CANN/BiSheng release and running the device
tests on both A2 and A3 hardware; a non-NPU development environment cannot
substitute for that verification.

The original project and retained CPU implementations remain subject to the
repository license and upstream attribution.
