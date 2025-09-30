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

import torch.nn as nn
from contextlib import contextmanager

model_configs = [
    './modelzoo/meta-llama/Llama-2-7b-hf',
    './modelzoo/meta-llama/Meta-Llama-3-8B',
    './modelzoo/meta-llama/Meta-Llama-3-8B-Instruct',
    './modelzoo/meta-llama/Llama-3.1-8B', 
    './modelzoo/meta-llama/Llama-3.1-8B-Instruct', 
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

@contextmanager
def silence_torch_module_repr():
    orig = nn.Module.__repr__
    try:
        nn.Module.__repr__ = lambda self: f"{self.__class__.__name__}()"
        yield
    finally:
        nn.Module.__repr__ = orig


def lm_eval_func(args, model, config_name):
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM
    
    model.eval()
    model = model.cuda()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(config_name, use_fast=False, use_auth_token=args.hf_token)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = args.tasks

    results = {}
    for task_name in task_names:
        print(f"Evaluating {task_name}...")
        result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size)['results']
        result = result[task_name]
        acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
        results[task_name] = acc
        print(f"acc: {acc}%")
    metric_vals = {task: result for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
    print(f"------------------------- ({config_name}) ------------------------")
    print(metric_vals)

    return metric_vals


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

def get_model_fp16(config_name):
    return modeling_llama.FlatQuantFP16LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2"
    )

def benchmark(args):
    final_results_fp16 = {}
    final_results_flat = {}
    for config_name in model_configs:
        pprint.pprint(vars(args))

        print(f'------------------------- FP16 ------------------------')
        # FP16
        args.fuseLN, args.trans = False, "none"
        args.online_trans = set()
        model = get_model_fp16(config_name)
        model.to('cuda')
        with silence_torch_module_repr():
            results = lm_eval_func(args, model, config_name)
        
        final_results_fp16[f"{config_name}"] = results
        del model
        _cleanup()
            
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
            with silence_torch_module_repr():
                results = lm_eval_func(args, model, config_name)
            
            final_results_flat[f"{config_name}"] = results
            del model
            _cleanup()

    for k, v in final_results_fp16.items():
        print(f'------------------------- FP16 {k}------------------------')
        print(v)

    for k, v in final_results_flat.items():
        print(f"------------------------- Flat {k}------------------------")
        print(v)

           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        help='Tasks to evaluate on LM Eval.')
    parser.add_argument('--lm_eval_batch_size', type=int, default=128, help='Batch size for evaluation with lm eval harness.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model access.')
    
    args = parser.parse_args()
    benchmark(args)