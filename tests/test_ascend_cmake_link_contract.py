from pathlib import Path

cmake = Path("CMakeLists.txt").read_text()

assert "add_library(gaffa_ascend_kernels STATIC" in cmake, (
    "AscendC kernels must live in a dedicated pure-ASC static target"
)
assert "${GAFFA_ASCEND_KERNEL_SOURCES}" in cmake
assert "add_library(gaffa_core STATIC\n  ${GAFFA_CORE_SOURCES}\n)" in cmake, (
    "gaffa_core must remain a CXX-only static target"
)

# CANN's documentation requires the AscendC host runtime libraries to be linked
# explicitly whenever the final link is performed by a non-ASC linker. GAFFA's
# final artifacts (pybind module, GTest executable, benchmarks) intentionally use
# the normal CXX linker, so model the CANN runtime as one transitive interface.
assert "add_library(gaffa_ascend_runtime INTERFACE)" in cmake
for library in [
    "ascendc_runtime",
    "runtime",
    "profapi",
    "unified_dlog",
    "mmpa",
    "ascend_dump",
    "c_sec",
    "error_manager",
    "ascendcl",
]:
    assert library in cmake, f"missing CANN AscendC host runtime dependency: {library}"

core_links = cmake.split("target_link_libraries(gaffa_core", 1)[1]
assert "gaffa_ascend_kernels" in core_links, (
    "gaffa_core must link the dedicated AscendC kernel target"
)
assert "gaffa_ascend_runtime" in core_links, (
    "gaffa_core must propagate the CANN AscendC host runtime to CXX consumers"
)
assert "target_compile_options(gaffa_ascend_kernels" in cmake
