#!/usr/bin/env bash

set -e

EXPECTED_CONDA_ENV="gaffa"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Error: CONDA_PREFIX is empty. Please run:"
  echo "  conda activate $EXPECTED_CONDA_ENV"
  return 1 2>/dev/null || exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]; then
  echo "Error: expected conda environment '$EXPECTED_CONDA_ENV', got '${CONDA_DEFAULT_ENV:-unknown}'. Please run:"
  echo "  conda activate $EXPECTED_CONDA_ENV"
  return 1 2>/dev/null || exit 1
fi

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
  x86_64|amd64)
    CONDA_TRIPLET="x86_64-conda-linux-gnu"
    ;;
  aarch64|arm64)
    CONDA_TRIPLET="aarch64-conda-linux-gnu"
    ;;
  *)
    echo "Error: unsupported host architecture: $HOST_ARCH"
    echo "Supported GAFFA development hosts are x86_64 and aarch64."
    return 1 2>/dev/null || exit 1
    ;;
esac

# Prefer the architecture-matched compiler installed by conda-forge's generic
# c-compiler/cxx-compiler packages. Fall back to already activated CC/CXX, then
# to the native system compiler so existing CANN developer images remain usable.
if [ -x "$CONDA_PREFIX/bin/${CONDA_TRIPLET}-gcc" ]; then
  export CC="$CONDA_PREFIX/bin/${CONDA_TRIPLET}-gcc"
elif [ -n "${CC:-}" ] && [ -x "$CC" ]; then
  export CC="$CC"
elif command -v gcc >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
else
  echo "Error: no usable C compiler was found for host architecture $HOST_ARCH."
  return 1 2>/dev/null || exit 1
fi

if [ -x "$CONDA_PREFIX/bin/${CONDA_TRIPLET}-g++" ]; then
  export CXX="$CONDA_PREFIX/bin/${CONDA_TRIPLET}-g++"
elif [ -n "${CXX:-}" ] && [ -x "$CXX" ]; then
  export CXX="$CXX"
elif command -v g++ >/dev/null 2>&1; then
  export CXX="$(command -v g++)"
else
  echo "Error: no usable C++ compiler was found for host architecture $HOST_ARCH."
  return 1 2>/dev/null || exit 1
fi

if [ -x "$CONDA_PREFIX/bin/${CONDA_TRIPLET}-gcov" ]; then
  export GCOV="$CONDA_PREFIX/bin/${CONDA_TRIPLET}-gcov"
elif command -v gcov >/dev/null 2>&1; then
  export GCOV="$(command -v gcov)"
else
  export GCOV=""
fi

export CONAN_HOME="$PWD/.conan2"
export GAFFA_ASCEND_ARCH="${GAFFA_ASCEND_ARCH:-dav-2201}"

# CANN 8.5+ uses <install-root>/cann/set_env.sh. Older toolkit installations
# may still expose ascend-toolkit/set_env.sh. An explicit GAFFA_CANN_SET_ENV
# always wins.
CANN_ENV=""
if [ -n "${GAFFA_CANN_SET_ENV:-}" ] && [ -f "$GAFFA_CANN_SET_ENV" ]; then
  CANN_ENV="$GAFFA_CANN_SET_ENV"
else
  candidates=(
    "/usr/local/Ascend/cann/set_env.sh"
    "$HOME/Ascend/cann/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "$HOME/Ascend/ascend-toolkit/set_env.sh"
  )

  if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    candidates+=(
      "$ASCEND_HOME_PATH/set_env.sh"
      "$(dirname "$ASCEND_HOME_PATH")/set_env.sh"
    )
  fi
  if [ -n "${ASCEND_CANN_PACKAGE_PATH:-}" ]; then
    candidates+=(
      "$ASCEND_CANN_PACKAGE_PATH/set_env.sh"
      "$(dirname "$ASCEND_CANN_PACKAGE_PATH")/set_env.sh"
    )
  fi

  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      CANN_ENV="$candidate"
      break
    fi
  done
fi

if [ -z "$CANN_ENV" ]; then
  echo "Error: CANN set_env.sh was not found."
  echo "Install a CANN development toolkit for Atlas A2/A3, then either:"
  echo "  source /usr/local/Ascend/cann/set_env.sh"
  echo "or set:"
  echo "  export GAFFA_CANN_SET_ENV=/path/to/cann/set_env.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$CANN_ENV"

# Some CANN releases expose the ASC CMake package through set_env.sh, while
# others document adding compiler/tikcpp/ascendc_kernel_cmake explicitly.
CANN_ROOT=""
if [ -n "${ASCEND_CANN_PACKAGE_PATH:-}" ] && [ -d "$ASCEND_CANN_PACKAGE_PATH" ]; then
  CANN_ROOT="$ASCEND_CANN_PACKAGE_PATH"
elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "$ASCEND_HOME_PATH" ]; then
  CANN_ROOT="$ASCEND_HOME_PATH"
else
  CANN_ROOT="$(dirname "$CANN_ENV")"
fi

ASC_CMAKE_DIR="$CANN_ROOT/compiler/tikcpp/ascendc_kernel_cmake"
if [ -d "$ASC_CMAKE_DIR" ]; then
  export CMAKE_PREFIX_PATH="$ASC_CMAKE_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi

if ! command -v bisheng >/dev/null 2>&1; then
  echo "Error: BiSheng compiler is not on PATH after sourcing: $CANN_ENV"
  echo "Verify that the CANN toolkit development components are installed."
  return 1 2>/dev/null || exit 1
fi

if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
  echo "Error: selected host compiler is not executable. Recreate environment.yml or check system GCC/G++."
  return 1 2>/dev/null || exit 1
fi

echo "HOST_ARCH=$HOST_ARCH"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CANN_ENV=$CANN_ENV"
echo "CANN_ROOT=$CANN_ROOT"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-<unset>}"
echo "ASC_CMAKE_DIR=$ASC_CMAKE_DIR"
echo "GAFFA_ASCEND_ARCH=$GAFFA_ASCEND_ARCH"
echo "CC=$CC"
echo "CXX=$CXX"
echo "GCOV=${GCOV:-<unavailable>}"
echo "CONAN_HOME=$CONAN_HOME"
echo

which python
python -m pip --version

echo
which bisheng
bisheng --version | head -n 1 || true

echo
"$CC" --version | head -n 1
"$CXX" --version | head -n 1

if command -v npu-smi >/dev/null 2>&1; then
  echo
  echo "npu-smi=$(command -v npu-smi)"
fi
