"""逐层激活量化敏感度探针 (FlatQuant 变换后)。
复用 main.py 的加载流程: get_model -> apply_flatquant -> (reload_matrix) -> reparameterize -> rtn_fwrd(W4固化)。
探针: 逐层开关 ActivationQuantizer.enable, 测输出 logits 相对「全 act 关闭(W4-only)」的 KL 散度。
三种扫法: A 单层 / B 前缀累积 / C 后缀累积。结果存 json。
"""
import os, json, torch
import torch.nn.functional as F
import transformers

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.flat_utils as flat_utils
import gptq_utils
from flatquant.quant_utils import ActivationQuantizer

PROBE_NSEG = int(os.environ.get("PROBE_NSEG", "8"))   # 探针用的 wikitext2 段数 (自检可小)
PROBE_TAG = os.environ.get("PROBE_TAG", "flatquant")   # 输出标签 (flatquant / rtn)

def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    trainloader = data_utils.get_loaders(args, args.cali_dataset, tokenizer,
                                         nsamples=args.nsamples, seqlen=model.seqlen, eval_mode=False)

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        if os.environ.get("NO_TRANS", "0") == "1":
            # RTN 基线: 把所有可学习变换设 None -> 纯 W4A4 量化, 无任何变换
            from flatquant.trans_utils import (SVDSingleTransMatrix, SVDDecomposeTransMatrix,
                                               InvSingleTransMatrix, InvDecomposeTransMatrix)
            TT = (SVDSingleTransMatrix, SVDDecomposeTransMatrix, InvSingleTransMatrix, InvDecomposeTransMatrix)
            n = 0
            for mod in model.modules():
                for name, child in list(mod.named_children()):
                    if isinstance(child, TT):
                        setattr(mod, name, None); n += 1
            logger.info(f"NO_TRANS: cleared {n} transform modules -> RTN baseline")
        elif args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        flat_utils.reparameterize_model(model)
        logger.info("reparameterized.")
    if args.w_bits < 16:
        gptq_utils.rtn_fwrd(model, utils.DEV, args) if not args.gptq else gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        logger.info("weight quantized (固化).")
    model.to(utils.DEV)

    layers = model.model.layers
    L = len(layers)
    logger.info(f"num layers = {L}")

    def set_act_quant(layer_ids):
        s = set(layer_ids)
        for i, layer in enumerate(layers):
            for m in layer.modules():
                if isinstance(m, ActivationQuantizer):
                    m.enable = (i in s)

    # 评测段 (wikitext2)
    testloader = data_utils.get_loaders(args, "wikitext2", tokenizer, seqlen=model.seqlen, eval_mode=True)
    ids = testloader.input_ids[:, :PROBE_NSEG * model.seqlen].to(utils.DEV)

    @torch.no_grad()
    def get_logits():
        outs = []
        for i in range(PROBE_NSEG):
            batch = ids[:, i*model.seqlen:(i+1)*model.seqlen]
            outs.append(model(batch).logits.float().cpu())
        return torch.cat(outs, dim=0)  # [nseg, seqlen, vocab]

    @torch.no_grad()
    def kl_to_ref(probe, ref):
        # 平均 KL(probe || ref) over tokens
        lp = F.log_softmax(probe, dim=-1)
        lr = F.log_softmax(ref, dim=-1)
        return (lp.exp() * (lp - lr)).sum(-1).mean().item()

    # 参考: 全 act 关闭 = W4-only
    set_act_quant([])
    ref = get_logits()
    logger.info("ref (W4-only, act off) collected.")

    # P0 自检: 全 act 开 (= W4A4 全量化) 的 ppl + KL
    set_act_quant(list(range(L)))
    full_kl = kl_to_ref(get_logits(), ref)
    full_ppl = eval_utils.ppl_eval(model, testloader)
    logger.info(f"[selfcheck] full-A4 ppl={full_ppl:.4f}  full-A4 KL_vs_W4only={full_kl:.5f}")

    res = {"tag": PROBE_TAG, "L": L, "nseg": PROBE_NSEG,
           "full_A4_ppl": full_ppl, "full_A4_kl": full_kl,
           "A_single": [], "B_prefix": [], "C_suffix": []}

    # A 单层隔离
    for i in range(L):
        set_act_quant([i]); res["A_single"].append(kl_to_ref(get_logits(), ref))
        logger.info(f"A i={i} KL={res['A_single'][-1]:.5f}")
    # B 前缀累积
    for n in range(1, L+1):
        set_act_quant(list(range(n))); res["B_prefix"].append(kl_to_ref(get_logits(), ref))
    # C 后缀累积
    for n in range(1, L+1):
        set_act_quant(list(range(L-n, L))); res["C_suffix"].append(kl_to_ref(get_logits(), ref))

    out = f"./outputs/sensitivity_{PROBE_TAG}.json"
    os.makedirs("./outputs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"saved {out}")

if __name__ == "__main__":
    main()
