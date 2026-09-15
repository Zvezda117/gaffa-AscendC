from pathlib import Path

path = Path("src/gaffa/ascend_kernels/preprocessing.asc")
assert path.exists(), "preprocessing.asc must exist"
text = path.read_text()
for symbol in [
    "gaffa_validate_finite_kernel",
    "gaffa_mean_scrunch_kernel",
    "gaffa_running_median_kernel",
    "gaffa_interpolate_subtract_kernel",
    "gaffa_normalise_kernel",
    "gaffa_launch_validate_finite_ascend",
    "gaffa_launch_mean_scrunch_ascend",
    "gaffa_launch_running_median_ascend",
    "gaffa_launch_interpolate_subtract_ascend",
    "gaffa_launch_normalise_ascend",
]:
    assert symbol in text, symbol
assert "DataCopyPad" in text
assert "TBuf<AscendC::TPosition::VECCALC>" in text
assert ".GetValue(" in text
assert ".SetValue(" not in text, "avoid GlobalTensor scalar stores / DCache hazards"
assert "__shared__" not in text
assert "__shfl" not in text
