import argparse
import gc
import pprint
import numpy as np
import torch
import time
import os

import deploy.transformers.modeling_llama as modeling_llama
import torch
import transformers
from safetensors.torch import load_file
import json

import flatquant.data_utils as data_utils
from tqdm import tqdm

model_configs = [
    "./modelzoo/llama-2-hf/llama-2-7b-hf",
    "./modelzoo/llama-3/llama-3-8b", 
    "./modelzoo/llama-3-instruct/llama-3-8b-instruct",
    "./modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct",
    "./modelzoo/llama-3.1/llama-3.1-8b"
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 2
num_bench_steps = 1

def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs, repeat_idx = i))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module, repeat_idx):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    _cleanup()
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


@torch.no_grad()
def ppl_eval(model, testenc):
    print('Evaluating ppl...')
    model.eval()
    model = model.cuda()
    max_length = 2048   # fix model max length

    testenc = testenc.input_ids
    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)

    # warmup
    for i in range(num_warmup_steps):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]
        out = model(batch)
    torch.cuda.synchronize()
    _cleanup()

    nlls = []
    inference_times = []
    
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        lm_logits = model(batch).logits

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        inference_times.append((end_time - start_time) * 1000)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * max_length): ((i + 1) * max_length)
        ][:, 1:].to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    return ppl.item(), avg_time, std_time

# Load from safetensors format
def load_from_safetensors(checkpoint_path):

    from safetensors import safe_open
    from safetensors.torch import load_file

    state_dict = {}
    
    if os.path.isdir(checkpoint_path):
        # Check for index file (sharded model)
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        single_path = os.path.join(checkpoint_path, "model.safetensors")
        
        if os.path.exists(index_path):
            # Sharded model - load index
            with open(index_path, 'r') as f:
                index = json.load(f)
            
            # Load all shards
            loaded_files = set()
            
            for tensor_name, filename in index["weight_map"].items():
                if filename not in loaded_files:
                    shard_path = os.path.join(checkpoint_path, filename)
                    with safe_open(shard_path, framework="pt") as f:
                        for key in f.keys():
                            if key in index["weight_map"] and index["weight_map"][key] == filename:
                                state_dict[key] = f.get_tensor(key)
                    loaded_files.add(filename)
            
        else:
            # Single file
            state_dict = load_file(single_path)
    
    # Reconstruct the checkpoint format from safetensors
    checkpoint = {
        "model_state_dict": {},
        "quantizers": {}
    }
    
    for k, v in state_dict.items():
        if k.startswith("quantizer."):
            parts = k.split(".")
            layer_name = ".".join(parts[1:-1])
            param_type = parts[-1]
            
            if param_type == "scale":
                if layer_name not in checkpoint["quantizers"]:
                    class Quantizer:
                        pass
                    checkpoint["quantizers"][layer_name] = Quantizer()
                checkpoint["quantizers"][layer_name].scale = v
        else:
            checkpoint["model_state_dict"][k] = v
    
    return checkpoint

# rename parameters for loading
def rename_keys(checkpoint):
    new_checkpoint = {
        "model_state_dict": {},
        "quantizers": {}
    }
    for k, v in checkpoint["model_state_dict"].items():
        new_k = k.replace("q_proj.linear", "q_proj") \
                 .replace("q_proj.act_quantizer", "inp_trans_q") \
                 .replace("k_proj.linear", "k_proj") \
                 .replace("k_proj.act_quantizer", "inp_trans_k") \
                 .replace("v_proj.linear", "v_proj") \
                 .replace("v_proj.act_quantizer", "inp_trans_v") \
                 .replace("o_proj.linear", "o_proj.1") \
                 .replace("o_proj.act_quantizer", "o_proj_trans") \
                 .replace("ln_trans.matrix_left", "left_matrix") \
                 .replace("ln_trans.matrix_right", "right_matrix") \
                 .replace("ln_trans", "inp_trans_k") \
                 .replace("o_trans.matrix", "o_proj_trans.right_matrix") \
                 .replace("gate_proj.linear", "gate_proj") \
                 .replace("gate_proj.act_quantizer", "inp_trans_g") \
                 .replace("up_proj.linear", "up_proj") \
                 .replace("up_proj.act_quantizer", "inp_trans_u") \
                 .replace("down_proj.linear", "down_proj.2") \
                 .replace("down_proj.act_quantizer", "down_proj.0") \
                 .replace("down_trans.matrix_left", "down_proj.0.left_matrix") \
                 .replace("down_trans.matrix_right", "down_proj.0.right_matrix")\
                 .replace("down_trans", "down_proj.0") \
                 .replace("up_gate_trans.matrix_left", "left_matrix") \
                 .replace("up_gate_trans.matrix_right", "right_matrix") \
                 .replace("up_gate_trans", "inp_trans_g") \
                 .replace("k_cache_quantizer.clip", "kclip") \
                 .replace("v_cache_quantizer.clip", "vclip") \
                 .replace("kcache_trans.matrix", "trans_matrix_k") \
                 .replace("vcache_trans.matrix", "trans_matrix_v")
        new_checkpoint["model_state_dict"][new_k] = v
    
    for k, v in checkpoint["quantizers"].items():
        new_k = k.replace("linear", "weight_scales") \
                 .replace("mlp.down_proj.weight_scales", "mlp.down_proj.2.weight_scales") \
                 .replace("self_attn.o_proj.weight_scales", "self_attn.o_proj.1.weight_scales")
        new_checkpoint["quantizers"][new_k] = v.scale

    return new_checkpoint

def get_model_quantized(args, config_name, checkpoint_path = None):
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights():
        model = modeling_llama.FlatQuantLlamaForCausalLM(args=args, config=config)
    if checkpoint_path:
        checkpoint = load_from_safetensors(checkpoint_path)
        new_checkpoint = rename_keys(checkpoint = checkpoint)
        missing_keys_1, unexpected_keys_1 = model.load_state_dict(new_checkpoint["model_state_dict"], strict=False)
        missing_keys_2, unexpected_keys_2  = model.load_state_dict(new_checkpoint['quantizers'], strict = False)

        
        for layer in model.model.layers:    
            layer.self_attn.inp_trans_q.register_buffer("left_matrix", layer.self_attn.left_matrix)
            layer.self_attn.inp_trans_k.register_buffer("left_matrix", layer.self_attn.left_matrix)
            layer.self_attn.inp_trans_v.register_buffer("left_matrix", layer.self_attn.left_matrix)
            layer.self_attn.inp_trans_q.register_buffer("right_matrix", layer.self_attn.right_matrix)
            layer.self_attn.inp_trans_k.register_buffer("right_matrix", layer.self_attn.right_matrix)
            layer.self_attn.inp_trans_v.register_buffer("right_matrix", layer.self_attn.right_matrix)

            layer.mlp.inp_trans_u.register_buffer("left_matrix", layer.mlp.left_matrix)
            layer.mlp.inp_trans_u.register_buffer("right_matrix", layer.mlp.right_matrix)
            layer.mlp.inp_trans_g.register_buffer("left_matrix", layer.mlp.left_matrix)
            layer.mlp.inp_trans_g.register_buffer("right_matrix", layer.mlp.right_matrix)

        for name, module in model.named_modules():
            for attr_name in ['clip_factor_a_max', 'clip_factor_a_min']:
                if hasattr(module, attr_name):
                    attr_value = getattr(module, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        delattr(module, attr_name)
                        setattr(module, attr_name, attr_value.item())

        missing_keys_both = set(missing_keys_1) & set(missing_keys_2)
        print("success to load real weights")

    torch.set_default_dtype(dtype_old)
    return model


def get_model_hf(config_name):
    return transformers.LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )

def get_model_fp16(config_name):
    return modeling_llama.FlatQuantFP16LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )


def run_prefill(model, bsz, prefill_length):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input))


def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    out = model(test_input)
    past_key_values = out.past_key_values
    del out
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        for _ in range(decode_steps):
            model(next_input, past_key_values=past_key_values)
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_length + decode_steps
        out = model(test_input)
        for _ in range(decode_steps):
            model(next_input, past_key_values=out.past_key_values)
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    model = model.cuda()
    time_prefill, _ = run_prefill(model, bsz, prefill)
    _cleanup()
    if decode is not None:
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        time_e2e, _ = run_e2e(model, bsz, prefill, decode)
        _cleanup()
    else:
        time_decode = time_e2e = None
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

## print allocated gpu memory
def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"{prefix} GPU memory - Allocated: {allocated:.2f} GB")
    return allocated

def benchmark(args):
    for config_name in model_configs:
        pprint.pprint(vars(args))

        print("Loading dataset...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(config_name, use_fast=False)
        test_data = data_utils.get_loaders(
                    args = None,
                    name = "wikitext2",
                    tokenizer = tokenizer,
                    seqlen = 2048,
                    eval_mode = True
                )
        print(f"Loaded dataset")

        # FP16
        args.fuseLN, args.trans = False, "none"
        args.online_trans = set()
        model = get_model_fp16(config_name)
        model.to('cuda')

        if args.random_mode:
            time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
                model, args.batch_size, args.prefill_seq_len, args.decode_steps)

        print(f'------------------------- FP16 ------------------------')
        ppl_f16, time_f16, std_f16 = ppl_eval(model = model, testenc = test_data)
        del model
        _cleanup()

        if args.random_mode:
            print(f"Prefill time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
            if args.decode_steps is not None and args.decode_steps != 0:
                print(f"Decode time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
                print(f"E2E time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")

        print(f"test-inference time: {time_f16:.3f} +- {1.96 * std_f16:.3f}ms per sequence")
        print(f"Perplexity: {ppl_f16:.3f}")

        # Int4
        print(f'------------------------- Int4 ------------------------')
        args.fuseLN, args.trans = False, "none"
        args.online_trans = set()
        model = get_model_quantized(args, config_name)

        if args.random_mode:
            time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark, mem_i4 = run_all_for_model(
                model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        
        ppl_i4_benchmark, time_i4_benchmark, std_i4_benchmark = ppl_eval(model = model, testenc = test_data)
        del model
        _cleanup()

        if args.random_mode:
            print_e2e_time(args, time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark,
                        time_prefill_f16, time_decode_f16, time_e2e_f16)
        
        speedup_i4_benchmark = time_f16 / time_i4_benchmark
        print(f"test-inference time: {time_i4_benchmark:.3f} +- {1.96 * std_i4_benchmark:.3f}ms per sequence")
        print(f"Speedup: {speedup_i4_benchmark:.3f}x")
        print(f"Perplexity: {ppl_i4_benchmark:.3f}")
        print(f"Perplexity degradation: {ppl_i4_benchmark / ppl_f16:.3f}")

        # QuaRot
        args.fuseLN, args.trans = True, "had"
        online_trans_list = [
            {"o_proj", "qk", "down_proj"}
        ]
        for online_trans in online_trans_list:
            print(f"------------------------- Int4 QuaRot ({'+'.join(online_trans)}) ------------------------")
            args.online_trans = online_trans
            model = get_model_quantized(args, config_name)

            if args.random_mode:
                time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
                    model, args.batch_size, args.prefill_seq_len, args.decode_steps)
            
            ppl_i4, time_i4, std_i4 = ppl_eval(model = model, testenc = test_data)
            del model
            _cleanup()

            if args.random_mode:
                print_e2e_time(args, time_prefill_i4, time_decode_i4, time_e2e_i4,
                            time_prefill_f16, time_decode_f16, time_e2e_f16,
                            time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark)
   
        speedup_i4 = time_f16 / time_i4
        print(f"test-inference time: {time_i4:.3f} +- {1.96 * std_i4:.3f}ms per sequence")
        print(f"Speedup: {speedup_i4:.3f}x Speedup loss: {(speedup_i4_benchmark - speedup_i4):.3f}")
        print(f"Perplexity: {ppl_i4:.3f}")
        print(f"Perplexity degradation: {ppl_i4 / ppl_f16:.3f}")
            
        # FlatQuant
        args.fuseLN, args.trans = False, "matmul"
        online_trans_list = [
            {"qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"}
        ]
        for online_trans in online_trans_list:
            print(f"------------------------- Int4 FlatQuant ({'+'.join(online_trans)}) ------------------------")
            args.online_trans = online_trans
            import os
            model_name = os.path.basename(config_name)
            weight_dir = os.path.join(".", "outputs", model_name, "w4a4", "exp")

            model = get_model_quantized(args, config_name, weight_dir)

            if args.random_mode:
                time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
                    model, args.batch_size, args.prefill_seq_len, args.decode_steps)
                        
            ppl_i4, time_i4, std_i4 = ppl_eval(model = model, testenc = test_data)
            del model
            _cleanup()

            if args.random_mode:
                print_e2e_time(args, time_prefill_i4, time_decode_i4, time_e2e_i4,
                            time_prefill_f16, time_decode_f16, time_e2e_f16,
                            time_prefill_i4_benchmark, time_decode_i4_benchmark, time_e2e_i4_benchmark)
            
            speedup_i4 = time_f16 / time_i4
            print(f"test-inference time: {time_i4:.3f} +- {1.96 * std_i4:.3f}ms per sequence")
            print(f"Speedup: {speedup_i4:.3f}x Speedup loss: {(speedup_i4_benchmark - speedup_i4):.3f}")
            print(f"Perplexity: {ppl_i4:.3f}")
            print(f"Perplexity degradation: {ppl_i4 / ppl_f16:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int,
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
    parser.add_argument(
        '--random_mode', action = 'store_true',
        help = 'Check speedup with random weight (original version)',
        default = False, 
    )
    
    args = parser.parse_args()
    if args.batch_size is None:
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            args.batch_size = bsz
            benchmark(args)
    else:
        benchmark(args)