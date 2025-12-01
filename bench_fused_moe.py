# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import itertools
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.int4_utils import *
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx
import argparse
import pandas as pd
import torch.profiler as profiler
from torch.profiler import tensorboard_trace_handler

from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
    get_block_size_M,
)


from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")


def generate_data(total_token, num_global_expert, num_local_expert, model_dim, topk, dtype):
    total_x = torch.randn((total_token, model_dim), dtype=dtype)
    score = torch.randn((total_token, num_global_expert), dtype=dtype)
    from aiter.fused_moe import fused_topk
    total_topk_weights, total_topk_ids = fused_topk(total_x, score, topk, True)
    local_expert_mask = torch.any(total_topk_ids < num_local_expert, dim=1)
    selected_indices = torch.nonzero(local_expert_mask, as_tuple=True)[0]
    topk_ids = total_topk_ids[selected_indices]
    input_x = total_x[selected_indices]
    topk_weights = total_topk_weights[selected_indices]
    return input_x, topk_ids, topk_weights


# input_x, topk_ids, topk_weights = generate_data(1024, 256, 8, 7168, 8, torch.bfloat16)

@benchmark()
def test_fmoe(
    dtype,
    bs_per_rank,
    model_dim,
    inter_dim,
    num_global_expert,
    num_local_expert,
    topk,
    actType,
    qType,
    AQDType,
    WQDType,
    use_g1u1=False,
    doweight_stage1=False,
    torch_profile=False,
):
    if get_gfx() not in ["gfx950"] and qType == aiter.QuantType.per_1x32:
        return
    torch_quant = aiter.get_torch_quant(qType)
    torch_act = aiter.get_torch_act(actType)
    ep_size = num_global_expert // num_local_expert
    total_token = bs_per_rank * ep_size
    input_x, topk_ids, topk_weights = generate_data(total_token, num_global_expert, num_local_expert, model_dim, topk, dtype)
    
    token = input_x.shape[0]

    if use_g1u1:
        w1 = torch.randn((num_local_expert, inter_dim * 2, model_dim), dtype=dtype)
    else:
        w1 = torch.randn((num_local_expert, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((num_local_expert, model_dim, inter_dim), dtype=dtype)
    
    expert_mask = torch.zeros(size=(num_global_expert,), dtype=torch.int32)
    expert_mask[:num_local_expert] = 1

    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(num_local_expert, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(num_local_expert, -1), quant_dtype=WQDType)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif qType == aiter.QuantType.per_128x128:

        def weight_per_128x128_quant(weight, quant_dtype):
            E, dim1, dim2 = weight.shape
            weight_blocks = weight.view(
                E, dim1 // 128, 128, dim2 // 128, 128
            )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_blocks = weight_blocks.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_blocks = weight_blocks.view(
                E, -1, 128 * 128
            )  # [E, num_blocks, 128*128]
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                E, dim1 // 128, dim2 // 128, 128, 128
            )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_qt = weight_qt.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
            weight_scale = weight_scale.view(
                E, dim1 // 128, dim2 // 128
            )  # [E, num_blocks_dim1, num_blocks_dim2]
            return weight_qt, weight_scale

        w1_qt, w1_scale = weight_per_128x128_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = weight_per_128x128_quant(w2, quant_dtype=WQDType)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)

    if qType != aiter.QuantType.per_1x32:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)

    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input_x.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    else:
        a1_qt, a1_scale = torch_quant(input_x, quant_dtype=AQDType)

    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w2_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)
            )
        )
    elif WQDType != dtypes.fp4x2:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

    # # ######################## fused 2 stage #########
    if dtype == dtypes.bf16:
        from aiter import QuantType, get_hip_quant

        if qType == aiter.QuantType.per_128x128:
            quant_func = get_hip_quant(QuantType.per_1x128)
        elif qType == aiter.QuantType.per_Tensor:
            quant_func = get_hip_quant(qType)
        else:
            raise RuntimeError
        random_padding = torch.randn((4096*16-input_x.shape[0], input_x.shape[1]), dtype=input_x.dtype, device=input_x.device)
        padded_tensor = torch.cat([input_x, random_padding], dim=0)
        a1q, a1q_scale = quant_func(padded_tensor, quant_dtype=AQDType)
        print(f'{w1_qt_aiter.shape=} {w1_qt_aiter.dtype=} {w1_scale.shape=} {w1_scale.dtype=}')
        out2_aiter, us_fuse = run_perftest(
            fused_moe,
            a1q,
            w1_qt_aiter,
            w2_qt_aiter,
            topk_weights,
            topk_ids,
            w1_scale=fp4_utils.e8m0_shuffle(
                w1_scale
            ),  # e8m0_shuffle will do nothing if it's a fp32
            w2_scale=fp4_utils.e8m0_shuffle(w2_scale),
            a1_scale=a1q_scale,
            quant_type=qType,
            activation=actType,
            doweight_stage1=doweight_stage1,
            expert_mask=expert_mask,
            num_local_tokens=torch.tensor([input_x.shape[0]], dtype=torch.int32),
            dtype=torch.bfloat16,
            needTrace=torch_profile,
        )
        w1_matmul_flops = model_dim * inter_dim * 2
        if use_g1u1:
            w1_matmul_flops *= 2
        w2_matmul_flops = inter_dim * model_dim * 2
        act_tokens = (topk_ids < num_local_expert).sum().item()
        total_flops = (w1_matmul_flops + w2_matmul_flops) * act_tokens
        actual_tflops = (total_flops / us_fuse) / 1e6

        return {"us": us_fuse, "act_tokens": act_tokens, "tflops": actual_tflops, 'm_per_expert': bs_per_rank * topk // num_local_expert}


l_dtype = ["bf16",]
l_dim = [(7168, 2048)]
l_bs_per_rank = [
    # 1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    # 4096,
    # 8192
]
l_quant = [
    # (aiter.QuantType.No, None, None),  # a16w16
    # (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    # (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    # (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    # (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
]
l_act = [aiter.ActivationType.Silu, aiter.ActivationType.Gelu][:1]
# l_doweight_stage1 = [False, True]
l_doweight_stage1 = [False]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8,  # a8w8""",
)

parser.add_argument(
    "-a",
    "--act",
    type=str,
    choices=["silu", "gelu"],
    default=None,
    help="""Select activation type.
    e.g.: -a silu""",
)

parser.add_argument(
    "-s",
    "--doweight_stage1",
    type=dtypes.str2bool,
    nargs="?",
    const=None,
    default=None,
    help="""Whether to do weight in stage 1. Default is [False, True].
    -s f    # False.
    -s t    # True.""",
)

parser.add_argument(
    "-e",
    "--num-local-expert",
    type=int,
    default=8,
    help="""Number of local experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "--num-global-expert",
    type=int,
    default=256,
    help="""Number of global experts.""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=8,
    help="""Number of top experts.
    e.g.: -k 8""",
)

parser.add_argument(
    "--profile",
    action="store_true",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

if args.dim is not None:
    l_dim = [args.dim]

if args.tokenNum is not None:
    l_bs_per_rank = [args.tokenNum]

l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant

if args.act is not None:
    l_act = [getattr(aiter.ActivationType, args.act.capitalize())]

if args.doweight_stage1 is not None:
    l_doweight_stage1 = [args.doweight_stage1]

for (
    dtype,
    act_type,
    (quant_type, aq_dtype, wq_dtype),
    (model_dim, inter_dim),
    doweight_stage1,
) in itertools.product(l_dtype, l_act, l_quant, l_dim, l_doweight_stage1):
    df = []
    for bs_per_rank in l_bs_per_rank:
        ret = test_fmoe(
            dtype,
            bs_per_rank,
            model_dim,
            inter_dim,
            args.num_global_expert,
            args.num_local_expert,
            args.topk,
            act_type,
            quant_type,
            aq_dtype,
            wq_dtype,
            use_g1u1=True,
            doweight_stage1=doweight_stage1,
            torch_profile=args.profile,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    # aiter.logger.info(f"summary:\n{df.to_string()}")
    df = df[['bs_per_rank', 'm_per_expert', 'us', 'act_tokens', 'tflops']]
    aiter.logger.info(f"summary:\n{df.to_string(index=False)}")
    ep_size = args.num_global_expert // args.num_local_expert
    df.to_csv(f'csv_ep{ep_size}.csv', index=False)
