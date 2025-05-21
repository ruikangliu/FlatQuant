import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from flatquant.quant_utils import WeightQuantizer, ActivationQuantizer
from flatquant.flat_utils import kronecker_matmul
from flatquant.trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from flatquant.function_utils import get_init_scale, get_decompose_dim

from deepseek_v3.kernel import act_quant, weight_dequant, fp8_gemm
from deepseek_v3.model import ModelArgs, apply_rotary_emb, MLA, MLP, MoE, RowParallelLinear, \
            gemm_impl, attn_impl, block_size
from deepseek_v3.model import linear as deepseek_linear


class FlatQuantizedLinear(nn.Module):
    def __init__(self, flat_args, linear, is_rowparallel=False, act_quantizer=None):
        super(FlatQuantizedLinear, self).__init__()
        self.flat_args = flat_args
        self.linear = linear
        self.is_rowparallel = is_rowparallel

        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(flat_args.w_bits, perchannel=True, sym=not(flat_args.w_asym), mse=False)
        self.act_quantizer = act_quantizer
        if self.act_quantizer is None:
            self.act_quantizer = ActivationQuantizer(bits=flat_args.a_bits, sym=not(flat_args.a_asym), lac=flat_args.lac, groupsize=flat_args.a_groupsize, )

        self.lwc = flat_args.lwc
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight

    def get_weight(self, ):
        if hasattr(self.linear.weight, "scale"):
            return weight_dequant(self.linear.weight, self.linear.weight.scale)
        else:
            return self.linear.weight

    def _ori_forward(self, hidden_states):
        if self.is_rowparallel:
            y = deepseek_linear(hidden_states, self.linear.weight)
            if self.world_size > 1:
                dist.all_reduce(y)
            if self.linear.bias is not None:
                y += self.linear.bias
        else:
            y = deepseek_linear(hidden_states, self.linear.weight, self.linear.bias)
        return y
        
    def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
        # weight = self.linear.weight.data
        weight = self.get_weight()
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        # learnable weight clipping 
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        
        # quantize weight
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)
        # quantize activation
        hidden_states = self.act_quantizer(hidden_states)

        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        if self.is_rowparallel:
            output = F.linear(hidden_states, weight)
            if self.world_size > 1:
                dist.all_reduce(output)
            if self.linear.bias is not None:
                output += self.linear.bias
        else:
            output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        if not self._eval_mode:
            return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans)
        else:
            return self._eval_forward(hidden_states)

    def _eval_forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        hidden_states = self.act_quantizer(hidden_states).to(x_dtype)

        output = self.linear(hidden_states)
        return output

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True


class FlatQuantMLA(nn.Module):
    def __init__(self, flat_args, module: MLA):
        super().__init__()
        self.flat_args = flat_args

        self.dim = module.dim
        self.n_heads = module.n_heads
        self.n_local_heads = module.n_local_heads
        self.q_lora_rank = module.q_lora_rank
        self.kv_lora_rank = module.kv_lora_rank
        self.qk_nope_head_dim = module.qk_nope_head_dim
        self.qk_rope_head_dim = module.qk_rope_head_dim
        self.qk_head_dim = module.qk_head_dim
        self.v_head_dim = module.v_head_dim
        self.softmax_scale = module.softmax_scale

        if attn_impl == "naive":
            self.register_buffer("k_cache", module.k_cache, persistent=False)
            self.register_buffer("v_cache", module.v_cache, persistent=False)
        else:
            self.register_buffer("kv_cache", module.kv_cache, persistent=False)
            self.register_buffer("pe_cache", module.pe_cache, persistent=False)

        if module.q_lora_rank == 0:
            self.wq = FlatQuantizedLinear(flat_args, module.wq)
        else:
            self.wq_a = FlatQuantizedLinear(flat_args, module.wq_a)
            self.q_norm = module.q_norm
            self.wq_b = FlatQuantizedLinear(flat_args, module.wq_b)

        self.wkv_a = FlatQuantizedLinear(flat_args, module.wkv_a)
        self.kv_norm = module.kv_norm
        # self.wkv_b = FlatQuantizedLinear(flat_args, module.wkv_b)
        self.wkv_b = module.wkv_b
        # self.wo = module.wo
        self.wo = FlatQuantizedLinear(flat_args, module.wo, is_rowparallel=True)

        self.add_fq_trans()
        self._ori_mode = False
        self._eval_mode = False

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # ---- here quant the q and kv_a ----
        if self._ori_mode:
            q, kv = self._ori_forward_qkv(x)
        else:
            q, kv = self._trans_forward_qkv(x)
        # ---- here quant the q and kv_a ----
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            raise NotImplementedError
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            kv = self.kv_norm(kv)
            if self._eval_mode:
                self.kv_cache[:bsz, start_pos:end_pos] = kv
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            if start_pos == 0:
                scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) +
                        torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.softmax_scale
            else:
                scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                        torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            raise NotImplementedError
        else:
            if start_pos == 0:
                x = torch.einsum("bsht,btc->bshc", scores, kv)
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        # ---- here quant the wo ----
        # x = self.wo(x.flatten(2))
        if self._ori_mode:
            x = self.wo._ori_forward(x.flatten(2))
        else:
            if self.wo_trans is not None:
                x = self.wo_trans(x.flatten(2))
            else:
                x = x.flatten(2)
            x = self.wo(x, qa_trans=self.wo_trans)
        # ---- here quant the wo ----
        return x

    def _trans_forward_qkv(self, x):
        if self.qkv_trans is not None:
            x = self.qkv_trans(x)
        if self.q_lora_rank == 0:
            q = self.wq(x, qa_trans=self.qkv_trans)
        else:
            q1 = self.wq_a(x, qa_trans=self.qkv_trans)
            q2 = self.q_norm(q1)
            if self.wqb_trans is not None:
                q3 = self.wqb_trans(q2)
            else:
                q3 = q2
            q = self.wq_b(q3, qa_trans=self.wqb_trans)
        
        kv = self.wkv_a(x, qa_trans=self.qkv_trans)
        return q, kv

    def _ori_forward_qkv(self, x):
        if self.q_lora_rank == 0:
            q = self.wq._ori_forward(x)
        else:
            q = self.wq_b._ori_forward(self.q_norm(self.wq_a._ori_forward(x)))
        kv = self.wkv_a._ori_forward(x)
        return q, kv

    def add_fq_trans(self):
        SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.flat_args.w_bits < 16 or self.flat_args.a_bits < 16:
            qkv_dim_left, qkv_dim_right = get_decompose_dim(self.wkv_a.linear.weight.shape[1])
            self.qkv_trans = DecomposeTransMatrix(qkv_dim_left, qkv_dim_right, add_diag=self.flat_args.add_diag)
            
            if hasattr(self, "wq_b"):
                wq_b_dim_left, wq_b_dim_right = get_decompose_dim(self.wq_b.linear.weight.shape[1])
                self.wqb_trans = DecomposeTransMatrix(wq_b_dim_left, wq_b_dim_right, add_diag=self.flat_args.add_diag)
            else:
                self.wqb_trans = None

            self.wkvb_trans = None
            # NOTE: no quantization for kv_b

            wo_dim_left, wo_dim_right = get_decompose_dim(self.wo.linear.weight.shape[1])
            self.wo_trans = DecomposeTransMatrix(wo_dim_left, wo_dim_right, add_diag=self.flat_args.add_diag)
        else:
            self.qkv_trans, self.wqb_trans, self.wkvb_trans, self.wo_trans = None, None, None, None

    def reparameterize(self, ):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
        if self.wqb_trans is not None:
            self.wqb_trans.to_eval_mode()
        if self.wo_trans is not None:
            self.wo_trans.to_eval_mode()
        if self.wkvb_trans is not None:
            self.wkvb_trans.to_eval_mode()

    def rep_matrix_only(self, ):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
        if self.wqb_trans is not None:
            self.wqb_trans.to_eval_mode()
        if self.wo_trans is not None:
            self.wo_trans.to_eval_mode()
        if self.wkvb_trans is not None:
            self.wkvb_trans.to_eval_mode()

    def clear_trans(self, ):
        self.qkv_trans, self.wqb_trans, self.wkvb_trans, self.wo_trans = None, None, None, None

class FlatQuantMLP(nn.Module):

    def __init__(self, flat_args, module: MLP):
        super().__init__()
        self.flat_args = flat_args
        self.w1 = FlatQuantizedLinear(flat_args, module.w1)
        self.w2 = FlatQuantizedLinear(flat_args, module.w2, is_rowparallel=True)
        self.w3 = FlatQuantizedLinear(flat_args, module.w3)
        self.add_fq_trans()

        self._ori_mode = False

    def _trans_forward(self, x):
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.w3(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.w1(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = F.silu(gate_states) * up_states
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.w2(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        '''origin implement: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        return self.w2._ori_forward(F.silu(self.w1._ori_forward(x)) * self.w3._ori_forward(x))

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def add_fq_trans(self):
        DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.flat_args.w_bits < 16 or self.flat_args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.w3.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.flat_args.add_diag)
            down_dim_left, down_dim_right = get_decompose_dim(self.w2.linear.weight.shape[1])
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.flat_args.add_diag)
        else:
            self.up_gate_trans, self.down_trans = None, None

    def reparameterize(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()

    def rep_matrix_only(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()

    def clear_trans(self, ):
        self.up_gate_trans, self.down_trans = None, None

class FlatQuantMoEExpert(nn.Module):
    def __init__(self, flat_args, module, act_quantizer_w1=None, act_quantizer_w2=None):
        super().__init__()

        self.w1 = FlatQuantizedLinear(flat_args, module.w1, act_quantizer=act_quantizer_w1)
        self.w2 = FlatQuantizedLinear(flat_args, module.w2, is_rowparallel=isinstance(module.w2, RowParallelLinear), act_quantizer=act_quantizer_w2)
        self.w3 = FlatQuantizedLinear(flat_args, module.w3, act_quantizer=act_quantizer_w1)

    def _ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2._ori_forward(F.silu(self.w1._ori_forward(x)) * self.w3._ori_forward(x))

    def _trans_forward(self, x_ts, w1_trans, w2_trans):
        up_states = self.w3(x_ts, qa_trans=w1_trans)
        gate_states = self.w1(x_ts, qa_trans=w1_trans)

        x_act_fn = F.silu(gate_states) * up_states

        if w2_trans is not None:
            x_ts_2 = w2_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.w2(x_ts_2, qa_trans=w2_trans)
        return down_states

    def forward(self, x, w1_trans, w2_trans):
        return self._trans_forward(x, w1_trans, w2_trans)


class FlatQuantMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, flat_args, module: MoE):

        super().__init__()
        self.flat_args = flat_args

        self.dim = module.dim
        self.n_routed_experts = module.n_routed_experts
        self.n_local_experts = module.n_local_experts
        self.n_activated_experts = module.n_activated_experts
        self.experts_start_idx = module.experts_start_idx
        self.experts_end_idx = module.experts_end_idx
        self.gate = module.gate
        self.act_quantizer_w1_routed = ActivationQuantizer(bits=flat_args.a_bits, sym=not(flat_args.a_asym), lac=flat_args.lac, groupsize=flat_args.a_groupsize, )
        self.act_quantizer_w2_routed = ActivationQuantizer(bits=flat_args.a_bits, sym=not(flat_args.a_asym), lac=flat_args.lac, groupsize=flat_args.a_groupsize, )
        self.experts = nn.ModuleList([FlatQuantMoEExpert(flat_args, module.experts[i], act_quantizer_w1=self.act_quantizer_w1_routed, act_quantizer_w2=self.act_quantizer_w2_routed) if module.experts[i] is not None else None for i in range(self.n_routed_experts)])
        
        self.shared_experts = FlatQuantMoEExpert(flat_args, module.shared_experts)
        self.add_fq_trans()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._ori_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        if not self._ori_mode and self.w1_trans is not None:
            x = self.w1_trans(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if self._ori_mode:
                y[idx] += expert._ori_forward(x[idx]) * weights[idx, top, None]
            else:
                # independent_w2_trans = True
                independent_w2_trans = False
                if independent_w2_trans:
                    y[idx] += expert(x[idx], self.w1_trans, self.routed_w2_trans[i]) * weights[idx, top, None]
                else:
                    y[idx] += expert(x[idx], self.w1_trans, self.routed_w2_trans) * weights[idx, top, None]
        if self._ori_mode:
            z = self.shared_experts._ori_forward(x)
        else:
            z = self.shared_experts(x, self.w1_trans, self.w2_trans)
        if self.world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

    def add_fq_trans(self):
        DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.flat_args.w_bits < 16 or self.flat_args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.shared_experts.w1.linear.weight.shape[1])
            self.w1_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.flat_args.add_diag)
            down_dim_left, down_dim_right = get_decompose_dim(self.shared_experts.w2.linear.weight.shape[1])
            self.w2_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.flat_args.add_diag)
            routed_down_dim_left, routed_down_dim_right = get_decompose_dim(self.experts[self.experts_start_idx].w2.linear.weight.shape[1])
            # independent_w2_trans = True
            independent_w2_trans = False
            if independent_w2_trans:
                self.routed_w2_trans = nn.ModuleList([DecomposeTransMatrix(routed_down_dim_left, routed_down_dim_right, add_diag=self.flat_args.add_diag) if self.experts[i] is not None else None for i in range(self.n_routed_experts)])
            else:
                self.routed_w2_trans = DecomposeTransMatrix(routed_down_dim_left, routed_down_dim_right, add_diag=self.flat_args.add_diag)
        else:
            self.w1_trans, self.w2_trans, self.routed_w2_trans = None, None, None


    def reparameterize(self, ):
        self.rep_matrix_only()

    def rep_matrix_only(self, ):
        if self.w1_trans is not None:
            self.w1_trans.to_eval_mode()
            self.w2_trans.to_eval_mode()
            # independent_w2_trans = True
            independent_w2_trans = False
            if independent_w2_trans:
                for trans in self.routed_w2_trans:
                    if trans is not None:
                        trans.to_eval_mode()
            else:
                self.routed_w2_trans.to_eval_mode()

    def clear_trans(self, ):
        self.w1_trans, self.w2_trans, self.routed_w2_trans = None, None, None