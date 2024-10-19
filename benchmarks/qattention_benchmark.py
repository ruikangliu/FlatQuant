import argparse
import pprint
import numpy as np
import torch
import time

from deploy.transformers.kv_cache import MultiLayerPagedKVCache4Bit


model_sizes = [
    (1, 32, 128), # llama-7b
    (1, 40, 128), # llama-13b
    (1, 64, 128)  # llama-70b   
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000 / num_bench_steps, memory_usage


def quantized_kv_cache_decode(
    n_layers, num_heads, head_dim, 
    batch_size, dtype, seq_len, 
    trans_dtype=torch.float16, trans="had"):
    device = torch.device("cuda:0")
    cache = MultiLayerPagedKVCache4Bit(
        batch_size=batch_size,
        page_size=seq_len, 
        max_seq_len=seq_len, 
        device=device, 
        n_layers=n_layers, # Ignornig n_layers as it does not affect speed
        num_heads=num_heads,
        head_dim=head_dim,
        disable_quant=dtype == torch.float16,
        trans_dtype=trans_dtype,
        trans=trans
    )
    query_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    key_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    value_states = torch.rand((batch_size, 1, num_heads, head_dim), device=device, dtype=torch.float16)
    def _fake_prefill_and_decode():
        cache._needs_init = [False] * len(cache._needs_init)
        cache.length = seq_len - 1
        forward_func = cache.update(key_states, value_states, layer_idx=0, cache_kwargs={})
        attn_out = forward_func(query_states)

    times = []
    for i in range(10):
        times.append(module_benchmark(_fake_prefill_and_decode))
    return zip(*times)


def qattention_benchmark(args):
    for n_layers, num_heads, head_dim in model_sizes:
        args.n_layers = n_layers
        args.num_heads = num_heads
        args.head_dim = head_dim

        time_fp16, memory_fp16 = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.bsz,
            dtype=torch.float16,
            seq_len=args.seq_len,
            trans_dtype=None
        )
        time_int4, memory_int4 = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.bsz,
            dtype="int4",
            seq_len=args.seq_len,
            trans_dtype=None
        )
        time_int4_had, _ = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.bsz,
            dtype="int4",
            seq_len=args.seq_len,
            trans_dtype=torch.float16
        )
        time_int4_inv, _ = quantized_kv_cache_decode(
            n_layers=n_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.bsz,
            dtype="int4",
            seq_len=args.seq_len,
            trans_dtype=torch.float16,
            trans="matmul"
        )

        pprint.pprint(vars(args))
        print(f"FP16 time: {np.mean(time_fp16):.3f} +- {1.96 * np.std(time_fp16):.3f}ms")
        print(f"Int4 time: {np.mean(time_int4):.3f} +- {1.96 * np.std(time_int4):.3f}ms "
              f"Speedup: {np.mean(time_fp16) / np.mean(time_int4):.3f}")
        print(f"Int4 (+had) time: {np.mean(time_int4_had):.3f} +- {1.96 * np.std(time_int4_had):.3f}ms "
              f"Speedup: {np.mean(time_fp16) / np.mean(time_int4_had):.3f}")
        print(f"Int4 (+inv) time: {np.mean(time_int4_inv):.3f} +- {1.96 * np.std(time_int4_inv):.3f}ms "
              f"Speedup: {np.mean(time_fp16) / np.mean(time_int4_inv):.3f}")
        
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
        qattention_benchmark(args)
    else:
        for bsz in [1, 2, 4, 8, 16, 32]:
            args.bsz = bsz
            qattention_benchmark(args)
