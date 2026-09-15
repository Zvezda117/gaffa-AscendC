from pathlib import Path

src = Path("src/gaffa/ascend_kernels/dedispersion.asc").read_text()
required = [
    'extern "C" __global__ __aicore__ void kernel_name',
    "gaffa_dedisperse_u8_kernel",
    "gaffa_dedisperse_u16_kernel",
    "gaffa_dedisperse_f32_kernel",
    "gaffa_launch_dedisperse_u8_ascend",
    "gaffa_launch_dedisperse_u16_ascend",
    "gaffa_launch_dedisperse_f32_ascend",
    "gaffa_launch_dedisperse_spectrum_u8_ascend",
    "gaffa_launch_dedisperse_spectrum_u16_ascend",
    "gaffa_launch_dedisperse_spectrum_f32_ascend",
    "GetBlockIdx()",
    "GetBlockNum()",
    "DataCopyPad",
    "GetValue",
]
for token in required:
    assert token in src, token
for forbidden in ["cuda", "__global__ void", "__shared__", "__shfl", "double ", "SetValue("]:
    assert forbidden not in src, forbidden
