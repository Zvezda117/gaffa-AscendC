from pathlib import Path

path = Path("src/gaffa/ascend_kernels/time_series.asc")
assert path.exists(), "time_series.asc must exist"
text = path.read_text()
assert "gaffa_downsample_weighted_sum_kernel" in text
assert "gaffa_convert_uint32_to_float_kernel" in text
assert "DataCopyPad" in text, "tail-safe GM output must use non-aligned DataCopyPad"
assert ".GetValue(" in text, "baseline scalar implementation must read GM safely"
assert ".SetValue(" not in text, "do not use GlobalTensor::SetValue / DCache scalar GM stores"
assert "static_cast<float>" in text, "uint32 conversion must preserve unsigned semantics"
