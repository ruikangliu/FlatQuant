import argparse
import gc
import functools
import pprint
import numpy as np
import torch
import time

import inference
import inference.transformers.modeling_llama as modeling_llama
import torch
import transformers


model_configs = [
    "./modelzoo/llama-2-7b",
    # "./modelzoo/llama-2-13b",
    # "./modelzoo/llama-3-8b",
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 3
num_bench_steps = 10


def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    torch.cuda.reset_max_memory_allocated()
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


def _build_cache(bsz, length, layer, disable_quant, num_key_value_heads, hidden_size, device, trans="had"):
    num_heads = num_key_value_heads
    model_dim = hidden_size
    # head_dim = model_dim // num_heads
    head_dim = 128  # TODO. fixed head dim for LLaMA models
    return inference.transformers.MultiLayerPagedKVCache4Bit(
        bsz=bsz,
        page_size=length, 
        max_seq_len=length, 
        device=device, 
        n_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        disable_quant=disable_quant,
        trans_dtype=None if disable_quant else torch.float16,
        trans=trans
    )


def get_model_quantized(args, config_name):
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = modeling_llama.FlatQuantLlamaForCausalLM(args=args, config=config)

    return model, functools.partial(
        _build_cache, 
        disable_quant=False,
        device=torch.device("cuda:0"),
        num_key_value_heads=model.config.num_key_value_heads,
        hidden_size=model.config.hidden_size,
        trans=args.trans if "qk" in args.online_trans else "none"), model.config.hidden_size


def get_model_fp16(config_name):
    model = modeling_llama.FlatQuantFP16LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )
    return model, functools.partial(
        _build_cache, 
        disable_quant=True,
        device=torch.device("cuda:0"),
        num_key_value_heads=model.config.num_key_value_heads,
        hidden_size=model.config.hidden_size,
    ), model.config.hidden_size


def run_prefill(layer, cache_builder, bsz, prefill_length, hidden_size):
    device = layer.self_attn.v_proj.weight.device
    test_input = torch.rand((bsz, prefill_length, hidden_size), dtype=torch.float16, device=device)
    if cache_builder is None:
        def _prefill():
            layer(test_input)
    else:
        past_key_values = cache_builder(bsz, prefill_length, layer)
        def _prefill():
            past_key_values.length = 0
            past_key_values._needs_init[0] = True
            layer(test_input, past_key_value=past_key_values)
    return module_benchmark(_prefill)


def run_decode(layer, cache_builder, bsz, prefill_length, decode_steps, hidden_size):
    device = layer.self_attn.v_proj.weight.device
    test_input = torch.rand((bsz, prefill_length, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((bsz, 1, hidden_size), dtype=torch.float16, device=device)
    assert cache_builder is not None
    past_key_values = cache_builder(bsz, prefill_length + decode_steps, layer)
    layer(test_input, past_key_value=past_key_values)
    def _decode_for_multiple_steps():
        past_key_values.length = prefill_length
        for i in range(decode_steps):
            layer(next_input, past_key_value=past_key_values, 
            position_ids=torch.tensor([[prefill_length + i]] * bsz, device=past_key_values.device, dtype=torch.int32))
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(layer, cache_builder, bsz, prefill_length, decode_steps, hidden_size):
    device = layer.self_attn.v_proj.weight.device
    test_input = torch.rand((bsz, prefill_length, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((bsz, 1, hidden_size), dtype=torch.float16, device=device)
    assert cache_builder is not None
    past_key_values = cache_builder(bsz, prefill_length + decode_steps, layer)
    def _prefill_and_decode_for_multiple_steps():
        past_key_values.length = 0
        past_key_values._needs_init[0] = True
        layer(test_input, past_key_value=past_key_values)
        for i in range(decode_steps):
            layer(next_input, past_key_value=past_key_values, 
            position_ids=torch.tensor([[prefill_length + i]] * bsz, device=device, dtype=torch.int32))
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


@torch.no_grad()
def run_all_for_model(layer, cache_builder, bsz, prefill, decode, hidden_size):
    layer = layer.cuda()
    layer.eval()
    time_prefill, _ = run_prefill(layer, cache_builder, bsz, prefill, hidden_size)
    
    _cleanup()
    if decode is not None and decode != 0:
        time_decode, memory_decode = run_decode(layer, cache_builder, bsz, prefill, decode, hidden_size)
        _cleanup()
        time_e2e, _ = run_e2e(layer, cache_builder, bsz, prefill, decode, hidden_size)
        _cleanup()
    else:
        time_decode = time_e2e = None
        memory_decode = None
    return time_prefill, time_decode, time_e2e, memory_decode


def print_e2e_time(args, time_prefill_i4, time_decode_i4, time_e2e_i4, time_prefill_f16, time_decode_f16, time_e2e_f16,
                   time_prefill_i4_benchmark=None, time_decode_i4_benchmark=None, time_e2e_i4_benchmark=None):
    prefill_speedup = np.mean(time_prefill_f16) / np.mean(time_prefill_i4)
    prefill_benchmark_speedup = np.mean(time_prefill_f16) / np.mean(time_prefill_i4_benchmark) if time_prefill_i4_benchmark is not None else None
    print(f"Prefill time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms\n"
            f"Speedup: {prefill_speedup:.3f}"
            + (f" Speedup loss: {prefill_benchmark_speedup - prefill_speedup:.3f}" if time_prefill_i4_benchmark is not None else ""))
    if args.decode_steps is not None and args.decode_steps != 0:
        decode_speedup = np.mean(time_decode_f16) / np.mean(time_decode_i4)
        decode_benchmark_speedup = np.mean(time_decode_f16) / np.mean(time_decode_i4_benchmark) if time_decode_i4_benchmark is not None else None
        print(f"Decode time: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms\n"
                f"Speedup: {decode_speedup:.3f}"
                + (f" Speedup loss: {decode_benchmark_speedup - decode_speedup:.3f}" if time_decode_i4_benchmark is not None else ""))
        e2e_speedup = np.mean(time_e2e_f16) / np.mean(time_e2e_i4)
        e2e_benchmark_speedup = np.mean(time_e2e_f16) / np.mean(time_e2e_i4_benchmark) if time_e2e_i4_benchmark is not None else None
        print(f"E2E time: {np.mean(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f}ms\n"
                f"Speedup: {e2e_speedup:.3f}"
                + (f" Speedup loss: {e2e_benchmark_speedup - e2e_speedup:.3f}" if time_e2e_i4_benchmark is not None else ""))


def benchmark(args):
    for config_name in model_configs:
        pprint.pprint(vars(args))

        # FP16
        args.fuseLN, args.trans = False, "none"
        args.online_trans = set()
        model, cache_builder, hidden_size = get_model_fp16(config_name)
        layer = model.model.layers[0]
        del model
        _cleanup()
        time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
            layer, cache_builder, args.bsz, args.prefill_seq_len, args.decode_steps, hidden_size)
        del layer
        _cleanup()
        print(f'------------------------- FP16 ------------------------')
        print(f"Prefill time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
        if args.decode_steps is not None and args.decode_steps != 0:
            print(f"Decode time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
            print(f"E2E time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")

        # Int4
        print(f'------------------------- Int4 ------------------------')
        args.fuseLN, args.trans = False, "none"
        args.online_trans = set()
        model, cache_builder, hidden_size = get_model_quantized(args, config_name)
        layer = model.model.layers[0]
        del model
        _cleanup()
        time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark, mem_i4 = run_all_for_model(
            layer, cache_builder, args.bsz, args.prefill_seq_len, args.decode_steps, hidden_size)
        del layer
        _cleanup()
        print_e2e_time(args, time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark,
                       time_prefill_f16, time_decode_f16, time_e2e_f16)

        # QuaRot
        args.fuseLN, args.trans = True, "had"
        online_trans_list = [
            {"o_proj", "qk", "down_proj"}
        ]
        for online_trans in online_trans_list:
            print(f"------------------------- Int4 QuaRot ({'+'.join(online_trans)}) ------------------------")
            args.online_trans = online_trans
            model, cache_builder, hidden_size = get_model_quantized(args, config_name)
            layer = model.model.layers[0]
            del model
            _cleanup()
            time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
                layer, cache_builder, args.bsz, args.prefill_seq_len, args.decode_steps, hidden_size)
            del layer
            _cleanup()
            print_e2e_time(args, time_prefill_i4, time_decode_i4, time_e2e_i4,
                           time_prefill_f16, time_decode_f16, time_e2e_f16,
                           time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark)
        
        # FlatQuant
        args.fuseLN, args.trans = False, "matmul"
        online_trans_list = [
            {"qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"}
        ]
        for online_trans in online_trans_list:
            print(f"------------------------- Int4 FlatQuant ({'+'.join(online_trans)}) ------------------------")
            args.online_trans = online_trans
            model, cache_builder, hidden_size = get_model_quantized(args, config_name)
            layer = model.model.layers[0]
            del model
            _cleanup()
            time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
                layer, cache_builder, args.bsz, args.prefill_seq_len, args.decode_steps, hidden_size)
            del layer
            _cleanup()
            print_e2e_time(args, time_prefill_i4, time_decode_i4, time_e2e_i4,
                           time_prefill_f16, time_decode_f16, time_e2e_f16,
                           time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=None,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        default=256,
    )
    
    args = parser.parse_args()
    if args.bsz is None:
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            args.bsz = bsz
            benchmark(args)
    else:
        benchmark(args)
