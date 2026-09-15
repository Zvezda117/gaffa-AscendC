from pathlib import Path

src = Path("src/gaffa/ascend_kernels/dedispersion_subband.asc").read_text()
required = [
    'extern "C" __global__ __aicore__ void kernel_name',
    "SubbandStage1Kernel",
    "SubbandStage2Kernel",
    "gaffa_subband_stage1_u8_kernel",
    "gaffa_subband_stage1_u16_kernel",
    "gaffa_subband_stage1_f32_kernel",
    "gaffa_subband_stage2_u32_kernel",
    "gaffa_subband_stage2_f32_kernel",
    "gaffa_launch_subband_stage1_u8_ascend",
    "gaffa_launch_subband_stage1_u16_ascend",
    "gaffa_launch_subband_stage1_f32_ascend",
    "gaffa_launch_subband_stage2_u32_ascend",
    "gaffa_launch_subband_stage2_f32_ascend",
    "GetBlockIdx()",
    "GetBlockNum()",
    "DataCopyPad",
    "GetValue",
]
for token in required:
    assert token in src, token
for forbidden in ["cuda", "__global__ void", "__shared__", "__shfl", "threadIdx", "blockIdx"]:
    assert forbidden not in src, forbidden
