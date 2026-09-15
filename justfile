set shell := ["bash", "-cu"]

export-env:
    conda env export --from-history > environment.lock.yml

verify-no-cuda:
    python tools/verify_no_cuda.py

conan-profile:
    source env/dev.sh && { test -f "$CONAN_HOME/profiles/default" || conan profile detect --force; }

deps: conan-profile
    source env/dev.sh && conan install . --build=missing -of build/conan/debug -s build_type=Debug -s compiler.cppstd=20

deps-release: conan-profile
    source env/dev.sh && conan install . --build=missing -of build/conan/release -s build_type=Release -s compiler.cppstd=20

install: deps
    source env/dev.sh && python -m pip install -e ".[dev]" --no-build-isolation --force-reinstall \
      -Cbuild-dir="build/editable-debug" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/debug/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/debug" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/debug" \
      -Ccmake.args="-DGAFFA_ASCEND_ARCH=${GAFFA_ASCEND_ARCH:-dav-2201}" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Debug"

test:
    source env/dev.sh && python -m pytest -v --cov

wheel: deps-release
    source env/dev.sh && python -m build --wheel --no-isolation \
      -Cbuild-dir="build/wheel-release" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/release/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/release" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/release" \
      -Ccmake.args="-DGAFFA_ASCEND_ARCH=${GAFFA_ASCEND_ARCH:-dav-2201}" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Release"

configure: deps
    source env/dev.sh && if [ -f build/dev/CMakeCache.txt ]; then \
      cmake -S . -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH"; \
    else \
      cmake -S . -B build/dev -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH"; \
    fi

reconfigure: deps
    source env/dev.sh && cmake --fresh -S . -B build/dev -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH"

build: configure
    source env/dev.sh && cmake --build build/dev

configure-release: deps-release
    source env/dev.sh && if [ -f build/release/CMakeCache.txt ]; then \
      cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH"; \
    else \
      cmake -S . -B build/release -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH"; \
    fi

build-release: configure-release
    source env/dev.sh && cmake --build build/release

build-benchmarks: configure-release
    source env/dev.sh && cmake --build build/release --target gaffa_benchmarks

bench-filterbank file="tests/data/basetest.fil" iterations="1": build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_filterbank_read_benchmark all {{file}} {{iterations}}

bench-spectrum-cpu file="tests/data/basetest.fil" dm="100" iterations="3" chan_begin="0" chan_end="0": build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_dedispersion_spectrum_cpu_benchmark {{file}} {{dm}} {{iterations}} {{chan_begin}} {{chan_end}}

bench-dedispersion-ascend file="tests/data/basetest.fil" ndm="32" dm_low="0" dm_step="1" iterations="5" device_id="0" subband_channels="32" ndm_per_nominal="32" time_tile_samples="81920": build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_dedispersion_ascend_benchmark {{file}} {{ndm}} {{dm_low}} {{dm_step}} {{iterations}} {{device_id}} {{subband_channels}} {{ndm_per_nominal}} {{time_tile_samples}}

bench-ffa-ascend nseries="8" nsamples="262144" iterations="5" device_id="0" tsamp="0.001" period_min="0.2" period_max="2.0": build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_ffa_ascend_benchmark {{nseries}} {{nsamples}} {{iterations}} {{device_id}} {{tsamp}} {{period_min}} {{period_max}}

test-cpp: build
    source env/dev.sh && ctest --test-dir build/dev --output-on-failure

# Device-backed Ascend tests are ordinary CTest cases and self-skip if no NPU
# is visible. On an A2/A3 runner this command exercises those cases as well.
test-ascend: test-cpp

coverage-cpp: deps
    source env/dev.sh && test -n "${GCOV:-}" && mkdir -p coverage && \
      cmake --fresh -S . -B build/coverage -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON \
        -DGAFFA_ASCEND_ARCH="$GAFFA_ASCEND_ARCH" && \
      cmake --build build/coverage --target clean && \
      { find build/coverage -name '*.gcda' -delete -o -name '*.gcno' -delete 2>/dev/null || true; } && \
      cmake --build build/coverage && \
      ctest --test-dir build/coverage --output-on-failure && \
      "$CONDA_PREFIX/bin/gcovr" \
        --gcov-executable "$GCOV" \
        --root . --filter src/gaffa --filter include/gaffa --exclude tests/cpp \
        --exclude src/gaffa/bindings.cpp --exclude 'src/gaffa/python/.*' \
        --exclude src/gaffa/io/filterbank_legacy.cpp \
        --txt --xml-pretty --xml coverage/cpp.xml \
        --html-details coverage/cpp.html

test-all: verify-no-cuda test test-cpp

clean:
    rm -rf build dist *.egg-info
