from pathlib import Path

kernel_dir = Path("src/gaffa/ascend_kernels")
kernels = sorted(kernel_dir.glob("*.asc"))
assert kernels, "AscendC kernel sources must exist"

for path in kernels:
    text = path.read_text()
    assert "reinterpret_cast<GM_ADDR>(" not in text, (
        f"{path}: BiSheng A2/A3 rejects host void* -> GM_ADDR reinterpret_cast; "
        "pass an ordinary uint8_t* into the GM_ADDR kernel launch instead"
    )

ffa = (kernel_dir / "ffa.asc").read_text()
for forbidden in [
    "static_cast<float>(head_rows",
    "static_cast<float>(tail_rows",
    "static_cast<float>(output_rows",
    "static_cast<float>(shift)",
    "RoundMode::CAST_TRUNC",
]:
    assert forbidden not in ffa, (
        f"ffa.asc: unsupported A2/A3 scalar conversion remains: {forbidden}"
    )
assert "RoundMode::CAST_FLOOR" in ffa, (
    "ffa.asc: positive round-half-up conversion must use supported CAST_FLOOR after +0.5"
)

series = (kernel_dir / "time_series.asc").read_text()
assert "static_cast<float>(input_gm_.GetValue" not in series, (
    "time_series.asc: uint32 GM values must not be cast directly to float on A2/A3"
)

pre = (kernel_dir / "preprocessing.asc").read_text()
for forbidden in [
    "static_cast<float>(factor_",
    "static_cast<float>(sample)",
    "static_cast<float>(nsamples_)",
    "static_cast<float>(lower)",
    "static_cast<std::uint32_t>(position)",
    "sqrtf(",
]:
    assert forbidden not in pre, (
        f"preprocessing.asc: unsupported A2/A3 scalar operation remains: {forbidden}"
    )
assert "AscendC::Rsqrt" in pre, (
    "preprocessing.asc: normalisation must use the A2/A3-supported vector Rsqrt API"
)
