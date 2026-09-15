from pathlib import Path

text = Path("src/gaffa/ascend_kernels/ffa.asc").read_text()
required = [
    "gaffa_ffa_detect_kernel",
    "gaffa_launch_ffa_detect_ascend",
    "GetBlockIdx",
    "GetBlockNum",
    "GetValue",
    "DataCopyPad",
]
for needle in required:
    assert needle in text, f"missing Ascend FFA detection contract symbol: {needle}"
for forbidden in ["threadIdx", "blockIdx", "__shared__", "__shfl", "atomicAdd", "cub::"]:
    assert forbidden not in text, f"CUDA-only primitive leaked into Ascend FFA kernel: {forbidden}"
