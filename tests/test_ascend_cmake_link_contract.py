from pathlib import Path

cmake = Path("CMakeLists.txt").read_text()

assert "add_library(gaffa_ascend_kernels STATIC" in cmake, (
    "AscendC kernels must live in a dedicated pure-ASC static target so CMake/ASC "
    "propagates the BiSheng default host runtime libraries to CXX consumers"
)
assert "${GAFFA_ASCEND_KERNEL_SOURCES}" in cmake
assert "add_library(gaffa_core STATIC\n  ${GAFFA_CORE_SOURCES}\n)" in cmake, (
    "gaffa_core must remain a CXX-only static target; mixing .asc sources into it "
    "causes the final CXX link to omit the ASC default runtime libraries"
)
assert "target_link_libraries(gaffa_core" in cmake
assert "gaffa_ascend_kernels" in cmake.split("target_link_libraries(gaffa_core", 1)[1], (
    "gaffa_core must link the dedicated AscendC kernel target"
)
assert "target_compile_options(gaffa_ascend_kernels" in cmake
