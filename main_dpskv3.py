import os
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from deepseek_v3.model import Transformer, ModelArgs, MLP, MoE

from contextlib import nullcontext

import time
import gc
import flatquant.utils as utils
import random

import logging
from termcolor import colored
from datetime import datetime

from typing import List
from deepseek_v3.eval_utils import ppl_eval
from flatquant.model_tools.deepseekv3_utils import FlatQuantMLP, FlatQuantMLA, FlatQuantMoE
from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:

    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def read_jsonl(file_name):
    print(f"load {file_name}")
    with open(file_name, mode='r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data


def get_self_infer_data(nsamples, seed, seqlen, tokenizer, json_path):
    traindata = read_jsonl(json_path)
    trainenc = []
    for item in traindata:
        message = [ {"role": "user", "content": f"{item['prompt']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}    ]
        prompt_tokens = tokenizer.apply_chat_template(message, add_generation_prompt=True)
        trainenc += prompt_tokens + tokenizer.encode(item['output'] + "\n\n")[1:]

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(trainenc) - seqlen - 1)
        j = i + seqlen
        inp = torch.tensor([trainenc[i:j]], dtype=torch.long)
        trainloader.append((inp, None))
    return trainloader


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    data_list = []
    with open("/home/ma-user/work/sunyuxuan/deepseek/datasets/wikitext/wikitext-2-raw/wiki.train.raw", encoding="utf-8") as f:
        for idx, row in enumerate(f):
            if row.strip():
                data_list.append(row.strip())
            else:
                data_list.append("")

        trainenc = tokenizer("\n\n".join(data_list), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def apply_flatquant_to_v3(args, model, apply_num=None):
    # Replace module with FlatQuant version
    if apply_num is None:
        apply_num = len(model.layers)
    for layer_id in range(apply_num):
        layer = model.layers[layer_id]
        # attn
        layer.attn = FlatQuantMLA(args, layer.attn)
        # mlp
        if isinstance(layer.ffn, MLP):
            layer.ffn = FlatQuantMLP(args, layer.ffn)
        elif isinstance(layer.ffn, MoE):
            layer.ffn = FlatQuantMoE(args, layer.ffn)
    return model


def reparameterize_model(model, apply_num=None):
    if apply_num is None:
        apply_num = len(model.layers)
    for layer_id in range(apply_num):
        layer = model.layers[layer_id]
        layer.attn.reparameterize()
        layer.ffn.reparameterize()
    return model


def load_flat_matrices(args, model, rank, path=None, apply_num=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_matrices_{rank}.pth"))
    if apply_num is None:
        apply_num = len(flat_parameters.keys())
    layers = model.layers
    for i in range(apply_num):
        flat_param = flat_parameters[i]
        layers[i].attn.rep_matrix_only()
        layers[i].ffn.rep_matrix_only()
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def load_flat_parameters(args, model, rank, path=None, apply_num=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters_{rank}.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_parameters_{rank}.pth"))
    if apply_num is None:
        apply_num = len(flat_parameters.keys())
    layers = model.layers
    for i in range(apply_num):
        flat_param = flat_parameters[i]
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def save_flat_matrices(args, model, rank=0, apply_num=None):
    flat_matrices = {}
    if apply_num is None:
        apply_num = len(model.layers)
    for i in range(apply_num):
        layer = model.layers[i]
        layer.self_attn.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        paras_name = ["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"]
        flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    torch.save(flat_matrices, os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth")))


def cali_flat_quant(args, model_args, model, dataloader, dev, rank, logger):
    model.eval()
    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP 
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        traincast = torch.cuda.amp.autocast

    # move embedding layer and first layer to target device
    layers = model.layers
    layers[0] = layers[0].to(dev)
    model.embed = model.embed.to(dev)
    model.freqs_cis = model.freqs_cis.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, args.cali_seqlen, model_args.dim), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, start_pos, freqs_cis, mask):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["start_pos"] = start_pos
            cache["freqs_cis"] = freqs_cis
            cache["mask"] = mask
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.embed = model.embed.cpu()
    torch.cuda.empty_cache()
    
    # same input of first layer for fp model and quant model
    fp_inps = inps   # take output of fp model as input
    fp_outs = torch.zeros_like(inps)

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    if args.v3_not_last:
        num_train_layer = len(layers) - 2
    else:
        num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        torch.distributed.barrier()
        logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        if args.deactive_amp:
            with torch.no_grad():
                layer.float() # NOTE: here

        torch.distributed.barrier()
        layer.attn._ori_mode = True
        layer.ffn._ori_mode = True
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), cache["start_pos"], cache["freqs_cis"], cache["mask"])
        layer.attn._ori_mode = False
        layer.ffn._ori_mode = False

        torch.distributed.barrier()
        logger.info(f"finished fp_outs")

        # TODO: add diag_scale in dpskv3 in the future
        layer = layer.to(dev)
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // args.cali_bsz), eta_min=args.flat_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        for epoch in range(args.epochs):
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    quant_out = layer(fp_inps[index:index+args.cali_bsz,], cache["start_pos"], cache["freqs_cis"], cache["mask"])
                    loss = loss_func(fp_outs[index:index+args.cali_bsz,], quant_out)
                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {i} lwc lac iter {epoch}, rank {rank}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}" )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters_{rank}.pth"))
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters_{rank}.pth")))
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    return model


def main(
    ckpt_path: str,
    config: str,
    args, 
):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    args.seed = 965

    utils.seed_everything(seed=args.seed)
    logger = create_logger(args.exp_dir)

    with open(config) as f:
        model_args = ModelArgs(**json.load(f))
    print(model_args)
    logger.info("---start to create model")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    logger.info("---- finished create tokenizer")

    if args.resume or args.reload_matrix:
        with torch.device("cuda"):
            model = Transformer(model_args)
        logger.info("---created model")
        
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        logger.info("---- finished load checkpoint")
        
        test_input = "蝙蝠侠是谁？"
        test_output = tokenizer.decode(generate(model, [tokenizer.encode(test_input)], 20, -1, -1)[0])
        print(test_output)
        logger.info(test_output)
        logger.info("---- finished test forward")
        ppl_fp = ppl_eval(model)
        logger.info(f"ppl_fp: {ppl_fp}")

        with torch.device("cuda"):
            apply_flatquant_to_v3(args, model)
        logger.info("---- finished apply_flatquant_to_v3/r1")
        if not args.reload_matrix:
            load_flat_parameters(args, model, rank)
        else:
            load_flat_matrices(args, model, rank, path=args.matrix_path)
        reparameterize_model(model)
        print(model)
        test_quant_output = tokenizer.decode(generate(model, [tokenizer.encode(test_input)], 20, -1, -1)[0])
        print(test_quant_output)
        ppl_quant = ppl_eval(model)
        logger.info(f"ppl_quant: {ppl_quant}")

    else:
        with torch.device("cpu"):
            model = Transformer(model_args)
        logger.info("---created model")

        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        logger.info("---- finished load checkpoint")

        if args.cali_dataset == 'wikitext2':
            dataloader = get_wikitext2(args.nsamples, args.seed, args.cali_seqlen, tokenizer, False)
            logger.info("---- finished load wikitest data")
        else:
            dataloader = get_self_infer_data(args.nsamples, args.seed, args.cali_seqlen, tokenizer, args.cali_json_path)
            logger.info("---- finished load self_infer data")
        logger.info("---- finished get data")
        if args.v3_not_last:
            apply_num = len(model.layers) - 2
        else:
            apply_num = None
        apply_flatquant_to_v3(args, model, apply_num=apply_num)

        logger.info("---- finished apply_flatquant_to_v3")
        print(model)

        cali_flat_quant(args, model_args, model, dataloader, 'cuda', rank, logger)

        if args.save_matrix:
            save_flat_matrices(args, model, rank)
        model = model.cuda()
        # reparameterize_model(model)
        # test_input = "蝙蝠侠是谁？"
        # test_quant_output = tokenizer.decode(generate(model, [tokenizer.encode(test_input)], 20, -1, -1)[0])
        # print(test_quant_output)
        logger.info("---- finished flat test forward")
        torch.distributed.barrier()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-path", type=str, default="./results")
    # General Arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model access.')
    parser.add_argument('--cali_seqlen', type=int, default=2048, help='seqlen for calibration.')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                                This applies to all linear layers in the model, including down-projection and out-projection.''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize.')
    parser.add_argument('--a_asym', action="store_true", default=False,
                        help='Use asymmetric activation quantization.')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the linear layers.')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize.')
    parser.add_argument('--w_asym', action="store_true", default=False,
                        help='Use asymmetric weight quantization.')

    # FlatQuant calibration Arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--cali_dataset', type=str, default='wikitext2',
                        help='Calibration dataset for FlatQuant.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for FlatQuant.')
    parser.add_argument("--cali_json_path", type=str, default="./a.json", help="Path to the self-infer data for the calibration of deepseek R1.")
    parser.add_argument('--cali_bsz', type=int, default=2,
                        help='Batch size for FlatQuant. Default is 1.')
    parser.add_argument("--flat_lr", type=float, default=1e-5, 
                        help='Learning rate for learnable transformation.')
    parser.add_argument("--cali_trans", default=False, action="store_true", 
                        help="Enable calibration of transformations.")
    parser.add_argument("--add_diag", default=False, action="store_true", 
                        help="Add per-channel scaling.")
    parser.add_argument("--lwc", default=False, action="store_true", 
                        help="Use learnable weight clipping.")
    parser.add_argument("--lac", default=False, action="store_true", 
                        help="Use learnable activation clipping.")
    parser.add_argument('--resume', action="store_true", default=False, 
                        help='Resume from a previous checkpoint for evaluation.')
    parser.add_argument('--save_matrix', action="store_true", default=False, 
                        help='Save the matrix-style parameters of FlatQuant.')
    parser.add_argument('--reload_matrix', action="store_true", default=False, 
                        help='Reload matrices and the inverse matrices for evaluation.')
    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Path to the pre-trained matrix-style parameters of FlatQuant.')
    parser.add_argument("--diag_init", type=str, default="sq_style", choices=["sq_style", "one_style"], 
                        help='The way to initialize per-channel scaling. Default is SmoothQuant style.')
    parser.add_argument("--diag_alpha", type=float, default=0.3, 
                        help='Hyperparameter for the SmoothQuant style initialization of per-channel scaling.')
    parser.add_argument("--warmup", default=False, action="store_true", help="Warm up the learning rate during training.")
    parser.add_argument("--deactive_amp", default=False, action="store_true", help="Disable AMP training.")
    parser.add_argument("--direct_inv", default=False, action="store_true", 
                        help="Use the inverse method in PyTorch to directly get the inverse matrix rather than SVD.")
    
    parser.add_argument('--v3_not_last', action="store_true", default=False, 
                        help='Not QUANT the last two layers of the deepseek V3/R1. ')
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory path.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")
    
    args = parser.parse_args()
    args.exp_dir = os.path.join(args.output_dir, "deepseek", f"w{args.w_bits}a{args.a_bits}", args.exp_name)
    
    main(args.ckpt_path, args.config, args)

