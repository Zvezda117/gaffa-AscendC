from pathlib import Path

src = Path("src/gaffa/ascend_kernels/ffa.asc").read_text()
required = [
    'extern "C" __global__ __aicore__ void gaffa_ffa_copy_level_kernel',
    'extern "C" __global__ __aicore__ void gaffa_ffa_merge_level_kernel',
    "gaffa_launch_ffa_copy_level_ascend",
    "gaffa_launch_ffa_merge_level_ascend",
    "FfaCopyLevelKernel",
    "FfaMergeLevelKernel",
    "GetBlockIdx()",
    "GetBlockNum()",
    "GetValue",
    "DataCopyPad",
]
for token in required:
    assert token in src, token
for forbidden in ["cuda", "__shared__", "__shfl", "threadIdx", "blockIdx"]:
    assert forbidden not in src, forbidden
