import torch
from inference.nn import Linear4bit, Quantizer
from inference.nn.online_trans import get_decompose_dim, OnlineTrans
import time
import argparse
import numpy as np
import pprint


# down_proj
mlp_sizes = [
    (4096, 4096),   # 7b/8b hidden size
    (5120, 5120),   # 13b hidden size
    (8192, 8192),   # 70b hidden size
    (11008, 4096),   # 2-7b down size
    (13824, 5120),  # 2-13b down size
    (14336, 4096),  # 3-8b down size
]

benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


@torch.inference_mode()
def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000 / num_bench_steps


@torch.inference_mode()
def linear4bit_benchmark(args):
    bsz = args.bsz
    seq_len = args.seq_len
    
    for (feature_dim_in, feature_dim_out) in mlp_sizes:
        for dtype in benchmark_dtypes:
            x = torch.rand((bsz,
                            seq_len,
                            feature_dim_in)).cuda().to(dtype)
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            int4_mod = torch.nn.Sequential(
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w)
            ).cuda()
            int4_mod_had = torch.nn.Sequential(
                OnlineTrans(baseline_mod.in_features, trans="had"),
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w),
            ).cuda()
            int4_mod_inv = torch.nn.Sequential(
                OnlineTrans(baseline_mod.in_features, trans="matmul"),
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w),
            ).cuda()

            pprint.pprint(vars(args))
            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")

            # fp16
            times_baseline = []
            for i in range(10):
                times_baseline.append(module_benchmark(baseline_mod, x))
            print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            # int4
            times = []
            for i in range(10):
                times.append(module_benchmark(int4_mod, x))
            print(f"Int4 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms\n"
                  f"Speedup: {np.mean(times_baseline) / np.mean(times):.3f}x")
            # int4_had
            times = []
            for i in range(10):
                times.append(module_benchmark(int4_mod_had, x))
            print(f"Int4 (+had) time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms\n"
                  f"Speedup: {np.mean(times_baseline) / np.mean(times):.3f}x")
            # int4_inv
            decompose_size = get_decompose_dim(feature_dim_in)[0]
            times = []
            for i in range(10):
                times.append(module_benchmark(int4_mod_inv, x))
            print(f"Int4 (+inv, {decompose_size}x{decompose_size} + {feature_dim_in // decompose_size}x{feature_dim_in // decompose_size}) time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms\n"
                f"Speedup: {np.mean(times_baseline) / np.mean(times):.3f}x")
            
            print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=None,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    
    args = parser.parse_args()
    if args.bsz is not None:
        linear4bit_benchmark(args)
    else:
        for bsz in [1, 2, 4, 8, 16, 32]:
            args.bsz = bsz
            linear4bit_benchmark(args)
