# Usage

for example
```
CUDA_VISIBLE_DEVICES=1 python bench_fused_moe.py -e 8 2>&1 | tee log_EP32.log
```

run `bash bench.sh` for all EP sizes.

add `--profile` if you want to have a profile file.
