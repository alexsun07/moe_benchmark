# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest, benchmark
from einops import rearrange
from einops import repeat as eirp
from aiter.ops.shuffle import shuffle_weight
import pandas as pd
import argparse

block_shape = (128, 128)


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(m, n, k):
    dtype = torch.bfloat16
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)

    return {
        # "dtype": weight.dtype,
        # "output dtype": b.dtype,
        "ck us": avg_b,
        "ck tflops": f'{m * n * k * 2 / avg_b / 1e6:.0f} TFLOPS',
        "ck BW": f'{(m * k + n * k + m * n) / avg_b / 1e3:.0f} GB/s',
    }


@perftest(num_iters=5)
def run_torch2(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


@perftest()
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    # return aiter.flatmm_a8w8_blockscale_ASM(x, weight, x_scale, w_scale, dtype)
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype, True)


@benchmark()
def test_gemm_asm_mi350(dtype, m, n, k):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    print("block_shape", block_shape)
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_k, scale_m], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_k, scale_n], dtype=dtypes.fp32, device="cuda")
    x_scale_trans = torch.transpose(x_scale, 0, 1)
    w_scale_trans = torch.transpose(w_scale, 0, 1).contiguous()
    flat_weight = shuffle_weight(weight, layout=(16, 16))
    a, avg_a = run_torch2(x, weight, x_scale_trans, w_scale_trans, dtypes.bf16)
    b, avg_b = run_asm(x, flat_weight, x_scale, w_scale_trans, dtype)
    a = a.to(dtypes.fp32)
    b = b.to(dtypes.fp32)
    tflops = 2 * m * n * k / (avg_b) / 1e6
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, asm avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}, tflops:{tflops:<8.2f}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=1e-2)


l_m = [
    # 64,
    # 128,
    # 256,
    # 512,
    # 1024,
    1,
    128,
    # 2048,
    4096,
    # 7188,
    8192,
    # 16384,
    # 32768,
]
l_nk = [
    # Qwen3-8B-FP8
    # (151936,4096),
    # (4096, 12288),
    # (6144, 4096),
    # (24576, 4096),
    # 32B-FP8
    (10240,5120),
    (5120,8192),
    (10240,5120),
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""N&K of mnk.
    e.g.: -nk 4096,512""",
)

args = parser.parse_args()
if args.m is not None:
    l_m = [args.m]
if args.nk is not None:
    l_nk = [args.nk]

def test_normal_gemm_a8w8_blockscale(l_mnk):
    df = []
    for m, n, k in l_mnk:
        ret = test_gemm(m, n, k)
        df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df.to_string()}")

l_quantDtype = ["fp8"]
l_mnk = []

for m in l_m:
    for n, k in l_nk:
        l_mnk.append((m, n, k))
# for m, n, k in l_mnk:
#     print(f'{m},{n},{k}')
test_normal_gemm_a8w8_blockscale(l_mnk)
