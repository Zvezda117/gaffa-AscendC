from pathlib import Path

path = Path("src/gaffa/ascend_kernels/time_series.asc")
assert path.exists(), "time_series.asc must exist"
text = path.read_text()
assert "gaffa_downsample_weighted_sum_kernel" in text
assert "gaffa_convert_uint32_to_float_kernel" in text
assert "DataCopyPad" in text, "tail-safe GM output must use non-aligned DataCopyPad"
assert ".GetValue(" in text, "baseline scalar implementation must read GM safely"
assert ".SetValue(" not in text, "do not use GlobalTensor::SetValue / DCache scalar GM stores"
assert "Uint32ToFloat" in text, "uint32-to-float conversion needs an A2/A3-safe helper"
assert "static_cast<std::int64_t>(value)" in text, (
    "uint32-to-float conversion must avoid the BiSheng-forbidden direct unsigned/float cast"
)
assert "static_cast<float>(input_gm_.GetValue" not in text
