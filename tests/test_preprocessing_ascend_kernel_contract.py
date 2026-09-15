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
    "Uint32ToFloat",
    "PositiveFloatFloorToUint32",
    "AscendC::ScalarCast",
    "AscendC::RoundMode::CAST_FLOOR",
    "AscendC::Sqrt",
]:
    assert symbol in text, symbol
assert "DataCopyPad" in text
assert "TBuf<AscendC::TPosition::VECCALC>" in text
assert ".GetValue(" in text
assert ".SetValue(" not in text, "avoid GlobalTensor scalar stores / DCache hazards"
assert "1.0F / stddev" in text, "normalisation must use reciprocal of precise Sqrt result"
for forbidden in [
    "reinterpret_cast<GM_ADDR>(",
    "AscendC::Rsqrt",
    "sqrtf(",
    "static_cast<float>(factor_",
    "static_cast<float>(sample)",
    "static_cast<float>(nsamples_)",
    "static_cast<float>(lower)",
    "static_cast<std::uint32_t>(position)",
    "__shared__",
    "__shfl",
]:
    assert forbidden not in text, forbidden
