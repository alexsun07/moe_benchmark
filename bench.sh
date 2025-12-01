CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 32 2>&1 | tee log_EP8.log
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 16 2>&1 | tee log_EP16.log
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 8 2>&1 | tee log_EP32.log
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 4 2>&1 | tee log_EP64.log
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 2 2>&1 | tee log_EP128.log
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 1 2>&1 | tee log_EP256.log
