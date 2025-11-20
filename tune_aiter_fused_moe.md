# How to tune aiter fused_moe

find `aiter/configs/untuned_fmoe.csv`

type the config you need. Please note that the token max number is 1024

for example

```csv
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
1,7168,2048,1,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,1,0
2,7168,2048,1,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,1,0
4,7168,2048,1,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,1,0
```

run `python3 hsa/gfx942/fmoe_2stages/tune.py`
