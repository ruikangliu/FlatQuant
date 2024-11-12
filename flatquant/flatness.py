import os
import math
import functools
from tqdm import tqdm

import torch
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from brokenaxes import brokenaxes

import flatquant.utils as utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.flat_utils as flat_utils
import flatquant.hadamard_utils as hadamard_utils
import flatquant.quant_utils as quant_utils


@torch.no_grad()
def get_act_stats(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    # activations
    inps = {}
    # weights
    weights = {}

    target_layer_type = torch.nn.Linear

    def stat_tensor(name, m, x, y):
        if not name in inps:
            inps[name] = []
            weights[name] = []
        w = m.weight
        inps[name].append(x.float().cpu())
        weights[name] = w.float().cpu()

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        stat_tensor(name, m, x, y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, target_layer_type) and not "lm_head" in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for data in dataset:
        model(data[0].to(device))
        break

    for name in list(inps.keys()):
        in_dim = inps[name][0].shape[-1]
        inps[name] = inps[name][0].reshape(-1, in_dim)

    for h in hooks:
        h.remove()

    return inps, weights


@torch.no_grad()
def get_flatness(args, logger, transform_type=None):
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    # apply pre-quantization transformations
    if transform_type is not None:
        if transform_type == "flatquant":
            args.w_bits = 4; args.a_bits = 4
            model = apply_flatquant_to_model(args, model)
            logger.info("Finished applying FlatQuant to model.")
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
            flat_utils.reparameterize_model(model)
            logger.info("Finished reparameterize model.")
            quant_utils.set_quantizer_state(model, enable=False)
        elif transform_type == "hadamard":
            pass
        elif transform_type == "smoothquant":
            args.act_scales_path = os.path.join(f"./act_scales/{args.model_name}.pt")
            args.act_scales = torch.load(args.act_scales_path)
            args.smooth_alpha = 0.85
        else:
            raise NotImplementedError        

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    inps, weights = get_act_stats(model, trainloader)
    flatness = {}
    for name in tqdm(inps.keys()):
        x = inps[name]
        w = weights[name]

        if transform_type is None or transform_type == "flatquant":
            x_flatness = LA.norm(x.cpu().numpy(), axis=0)
            w_flatness = LA.norm(w.cpu().numpy(), axis=0)
            flatness[name] = {
                "x": x_flatness, "w": w_flatness
            }
        elif transform_type == "hadamard":
            hidden_dim = x.shape[-1]
            if hadamard_utils.is_pow2(hidden_dim):
                had = hadamard_utils.get_had_pow2(hidden_dim).cuda()
                x_had = x.cuda() @ had
                w_had = w.cuda() @ had
            else:
                had_right, k = hadamard_utils.get_hadK(hidden_dim)
                had_right = had_right.cuda() / math.sqrt(k)
                had_left = hadamard_utils.get_had_pow2(hidden_dim // k).cuda()
                x_had = flat_utils.kronecker_matmul(x.cuda(), had_left, had_right)
                w_had = flat_utils.kronecker_matmul(w.cuda(), had_left, had_right)
            x_had_flatness = LA.norm(x_had.cpu().numpy(), axis=0)
            w_had_flatness = LA.norm(w_had.cpu().numpy(), axis=0)
            flatness[name] = {
                "x": x_had_flatness, "w": w_had_flatness
            }
        elif transform_type == "smoothquant":
            act_scales = args.act_scales[name].to(x.device)
            weight_scales = w.abs().max(dim=0)[0].clamp(min=1e-5)
            scales = (
                (act_scales.pow(args.smooth_alpha) / weight_scales.pow(1 - args.smooth_alpha))
                .clamp(min=1e-5)
            )
            x_sq = x / scales
            w_sq = w * scales
            x_sq_flatness = LA.norm(x_sq.cpu().numpy(), axis=0)
            w_sq_flatness = LA.norm(w_sq.cpu().numpy(), axis=0)
            flatness[name] = {
                "x": x_sq_flatness, "w": w_sq_flatness
            }

    args.cache_dir = os.path.join(args.vis_dir, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(flatness, os.path.join(args.cache_dir, f"flatness_{transform_type}.pt" if transform_type is not None else f"flatness.pt"))
    logger.info(f"Flatness stats saved at {args.cache_dir}.")

    del model
    utils.cleanup_memory()

    return flatness


def get_act_scales(args, logger):
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # get calibration data
    dataset = data_utils.get_loaders(
        args, args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    device = next(model.parameters()).device

    act_scales = {}

    target_layer_type = torch.nn.Linear

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, target_layer_type) and not "lm_head" in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(len(dataset))):
        input_ids = dataset[i][0].to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    args.act_scales_dir = "./act_scales"
    os.makedirs("./act_scales", exist_ok=True)
    torch.save(act_scales, os.path.join(args.act_scales_dir, f"{args.model_name}.pt"))
    logger.info(f"Activation scales saved at {args.act_scales_dir}.")

    del model
    utils.cleanup_memory()

    return act_scales


def plot_flatness(args, name, vectors, vector_names, y_max=None, y_broken=None, label_pad=15):
    for i in range(len(vectors)):
        sorted_indices = sorted(range(len(vectors[i])), key=lambda k: abs(vectors[i][k]), reverse=True)
        vectors[i] = [vectors[i][j] for j in sorted_indices]

    colors = ["#f44336", "#0288d1", "#8bc34a", "#808080"]
    fontsize = 20
    label_fontsize = 25
    linewidth = 4
    step_cnt = 100
    if y_broken is None:
        fig, ax1 = plt.subplots(1, 1)
    else:
        ax1 = brokenaxes(
            ylims=((0, y_broken[0]), (y_broken[1], y_max)),
            hspace=0.03,
        )

    # plot weight distribution
    for i in range(len(vectors)):
        x = np.linspace(0, len(vectors[i]) - 1, step_cnt)
        y = np.interp(x, range(len(vectors[i])), vectors[i])
        ax1.plot(x, y, color=colors[i], linewidth=linewidth, zorder=1000*(len(vectors) - i), label=vector_names[i])
    
    ax1.set_ylabel("Magnitude", fontsize=label_fontsize, labelpad=fontsize+label_pad if y_broken is not None else None)
    ax1.set_xlabel("Channels", fontsize=label_fontsize, labelpad=fontsize+8 if y_broken is not None else None)
    ax1.grid(axis='x', linestyle='--')
    ax1.grid(axis='y', linestyle='--')
    ax1.tick_params(axis="x", labelsize=fontsize-2)
    ax1.tick_params(axis="y", labelsize=fontsize-2)
    ax1.legend(loc="upper right", fontsize=fontsize-2)
    if y_broken is not None:
        for spine in ax1.spines["top"]:
            spine.set_visible(True)
        for spine in ax1.spines["right"]:
            spine.set_visible(True)
    elif y_max is not None:
        ax1.set_ylim(0, y_max)
    else:
        ax1.set_ylim(bottom=0)

    plt.tight_layout()
    if y_broken is not None:
        for handle in ax1.diag_handles:
            handle.remove()
        ax1.draw_diags()

    plt.savefig(os.path.join(args.vis_dir, f"{name}.pdf"))
    plt.close()


def plot_flatness_all_layers(args, flatness_flatquant, flatness_hadamard,
                             flatness_smoothquant, flatness_vanilla):
    for name in flatness_vanilla.keys():
        xw_flatnesses = [flatness_flatquant[name + ".linear"], flatness_hadamard[name], flatness_smoothquant[name], flatness_vanilla[name]]
        flatness_names = ["FlatQuant", "Hadamard", "SmoothQuant", "Vanilla"]
        x_flatnesses = [xw_flatness["x"] for xw_flatness in xw_flatnesses]
        w_flatnesses = [xw_flatness["w"] for xw_flatness in xw_flatnesses]
        plot_flatness(args, name + ".x", x_flatnesses, flatness_names)
        plot_flatness(args, name + ".w", w_flatnesses, flatness_names)
