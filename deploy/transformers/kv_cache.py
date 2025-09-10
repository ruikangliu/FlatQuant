from transformers.cache_utils import Cache
from typing import Optional, Tuple, Dict, Any
import math
import torch
from .. import _CUDA
import functools
from fast_hadamard_transform import hadamard_transform
from deploy.functional.quantization import get_minq_maxq


@torch.jit.script
def asym_quantize_and_pack_i4(x: torch.Tensor, clip_factor_a_max: torch.Tensor, clip_factor_a_min: torch.Tensor, lac: bool = False, quantize: bool = True):
    minq, maxq = get_minq_maxq(bits=4, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)

    if lac:
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        xmax = xmax * clip_factor_a_max
        xmin = xmin * clip_factor_a_min

        xmin_zero = torch.eq(xmin, 0.0)
        xmax_zero = torch.eq(xmax, 0.0)
        zero_mask = xmin_zero * xmax_zero
        
        neg_one = torch.full_like(xmin, -1.0)
        pos_one = torch.full_like(xmax, 1.0)
        
        xmin = torch.where(zero_mask, neg_one, xmin)
        xmax = torch.where(zero_mask, pos_one, xmax)

        scale = (xmax - xmin) / maxq
        zero = torch.round((-1.0 * xmin) / scale)
        q = torch.clamp((x / scale).round() + zero, 0, maxq)

        if not quantize:
            return scale * (q - zero), scale, zero

    else:
        scale = ((xmax - xmin).clamp(min=1e-5) / maxq)
        zero = -xmin
        q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

        if not quantize:
            return scale * (q - zero), scale, zero

    # pack int4
    q = q.to(dtype=torch.uint8)
    q = q[..., 0::2] | (q[..., 1::2] << 4)
    return q, scale, zero


def unpack_i4_and_asym_dequantize(q, scale, zero, lac: bool = False):
    #unpack int4
    assert q.dtype == torch.uint8
    q = torch.stack((q & 0x0f, (q >> 4) & 0x0f), dim=-1).view(*q.shape[:-1], q.shape[-1] * 2)
    if lac:
        return scale * (q - zero)
    else:
        return q * scale - zero


def matmul_had_cuda(X, dtype):
    n = X.shape[-1]
    input = hadamard_transform(X.to(dtype).contiguous(), scale=1/math.sqrt(n))
    return input.to(X.dtype).view(X.shape) 


def init_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _CUDA.init_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _CUDA.append_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)


def batch_decode_i4(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _CUDA.batch_decode_i4(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)


def init_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _CUDA.init_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _CUDA.append_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)


def batch_decode_f16(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _CUDA.batch_decode_f16(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)


class _AttentionStub(object):
    def __init__(self, cache_page_size, device, n_layers, disable_quant, trans_dtype, head_dim):
        self.cache_page_size = cache_page_size
        self.n_layers = n_layers
        self.disable_quant = disable_quant
        self.trans_dtype = trans_dtype
        self.head_dim = head_dim

    def forward(self, q, num_kv_heads, attention_kwargs, layer_idx):
        batch_size, q_len, num_qo_heads, head_dim = q.shape
        assert q_len == 1
        q = q.view(batch_size, num_qo_heads, head_dim)
        if self.trans_dtype is not None:
            if self.head_dim is None:
                q = matmul_had_cuda(q, dtype=self.trans_dtype) 
            else:
                q = torch.matmul(q.to(self.trans_dtype), self.head_dim)
        attn_output = torch.empty_like(q)
        if self.disable_quant:
            batch_decode = batch_decode_f16
        else:
            batch_decode = batch_decode_i4
        batch_decode(
            attn_output, q, 
            **attention_kwargs, layer_idx=layer_idx
        )
        attn_output = attn_output.unsqueeze(1)
        return attn_output


class MultiLayerPagedKVCache4Bit(Cache):
    def __init__(
        self, batch_size, page_size, max_seq_len, 
        device, n_layers, num_heads, head_dim, 
        disable_quant=False, trans_dtype=torch.float16,
        trans="had", group_size = 1):
        self.page_size = page_size
        self.batch_size = batch_size
        max_page_cnt = self.page_cnt_from_length(max_seq_len)
        self.disable_quant = disable_quant
        self.pages = torch.empty(
            (
                max_page_cnt * batch_size, 
                n_layers, 
                2, 
                num_heads, 
                page_size, 
                head_dim if disable_quant else head_dim // 2 
            ), 
            dtype=torch.float16 if disable_quant else torch.uint8, device=device)
        
        self.org_head_dim = head_dim
        self.trans = trans
        self.scales = torch.empty((max_page_cnt * batch_size, n_layers, 2, num_heads, page_size,  2), dtype=torch.float16, device=device)
        self.page_size = page_size
        self.max_seq_len = max_seq_len
        self._needs_init = [True] * n_layers
        self.length = 0
        self.device = device
        self.trans_dtype = trans_dtype
        self.n_layers = n_layers
        self.group_size = group_size

    def page_cnt_from_length(self, length):
        return (length + self.page_size - 1) // self.page_size
    
    def _ensure_page_cnt_per_batch(self, expected_page_cnt_per_batch):
        expected_total = expected_page_cnt_per_batch * self.batch_size
        current_total  = self.pages.shape[0]
        if expected_total <= current_total:
            return

        grow_to_total = max(expected_total, current_total * 2)
        add_total = grow_to_total - current_total

        new_pages = torch.empty(
            (add_total, *self.pages.shape[1:]),
            dtype=self.pages.dtype, device=self.pages.device
        )
        self.pages = torch.cat([self.pages, new_pages], dim=0)

        new_scales = torch.empty(
            (add_total, *self.scales.shape[1:]),
            dtype=self.scales.dtype, device=self.scales.device
        )
        self.scales = torch.cat([self.scales, new_scales], dim=0)

    @property
    def seen_tokens(self):
        return self.length
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        self.trans_matrix_k = cache_kwargs.get("trans_matrix_k")
        self.trans_matrix_k_inv_t = cache_kwargs.get("trans_matrix_k_inv_t")
        self.trans_matrix_v = cache_kwargs.get("trans_matrix_v")
        self.kclip_factor_a_max = cache_kwargs.get("kclip_factor_a_max")
        self.kclip_factor_a_min = cache_kwargs.get("kclip_factor_a_min")
        self.vclip_factor_a_max = cache_kwargs.get("vclip_factor_a_max")
        self.vclip_factor_a_min = cache_kwargs.get("vclip_factor_a_min")
        
        if self.trans == "had":
            self.head_dim = None
        elif self.trans.startswith("matmul"):
            self.head_dim = torch.randn([self.org_head_dim, self.org_head_dim], requires_grad=False).to(self.trans_dtype).to(self.device)
            self.head_dim = self.trans_matrix_k
            self.head_dim_inv_t = self.trans_matrix_k_inv_t
        else:
            self.trans_dtype = None
            self.head_dim = None
            
        self._stub = _AttentionStub(
            self.page_size, self.device, self.n_layers, 
            disable_quant=self.disable_quant, 
            trans_dtype=self.trans_dtype,
            head_dim = self.head_dim_inv_t if self.trans.startswith("matmul") else self.head_dim) # isFlatQ
        
        b_sz, added_length, num_heads, head_dim = key_states.shape

        orig_key_states = key_states
        orig_value_states = value_states
        if self.trans_dtype is not None:
            if self.head_dim is None:
                key_states = matmul_had_cuda(key_states, dtype=self.trans_dtype)
            else:
                key_states = torch.matmul(key_states.to(self.trans_dtype), self.head_dim) ## trans for k
        if self.disable_quant:
            k_scale = key_states.new_ones((b_sz, added_length, num_heads, 1))
            k_zero = key_states.new_zeros((b_sz, added_length, num_heads, 1))
            v_scale = value_states.new_ones((b_sz, added_length, num_heads, 1))
            v_zero = value_states.new_zeros((b_sz, added_length, num_heads, 1))
        else:
            self.kclip_factor_a_max, self.kclip_factor_a_min = torch.sigmoid(self.kclip_factor_a_max), torch.sigmoid(self.kclip_factor_a_min)
            self.vclip_factor_a_max, self.vclip_factor_a_min = torch.sigmoid(self.vclip_factor_a_max), torch.sigmoid(self.vclip_factor_a_min)
            if self.trans.startswith("matmul"):
                self.lac = True
            else:
                self.lac = False
            orig_key_states = key_states
            orig_value_states = value_states
            # key_states_lac_only, _, _ = asym_quantize_and_pack_i4(key_states, clip_factor_a_max = self.kclip_factor_a_max, clip_factor_a_min = self.kclip_factor_a_min, lac = self.lac, quantize = False) # if we wants to use lac for prefill kv, use it
            # value_states_lac_only, _, _ = asym_quantize_and_pack_i4(value_states, clip_factor_a_max = self.vclip_factor_a_max, clip_factor_a_min = self.vclip_factor_a_min, lac = self.lac, quantize = False)
            key_states, k_scale, k_zero = asym_quantize_and_pack_i4(key_states, clip_factor_a_max = self.kclip_factor_a_max, clip_factor_a_min = self.kclip_factor_a_min) # lac = false / if we wnats to use lac for cached kv, use lac = True
            value_states, v_scale, v_zero = asym_quantize_and_pack_i4(value_states, clip_factor_a_max = self.vclip_factor_a_max, clip_factor_a_min = self.vclip_factor_a_min)
        
        if self.group_size > 1:
            # Repeat the quantized states
            key_states = key_states.repeat_interleave(self.group_size, dim = 2)
            value_states = value_states.repeat_interleave(self.group_size, dim = 2)
            # Update num_heads to match after repeat
            num_heads = num_heads * self.group_size
        
        k_param = torch.cat([k_scale, k_zero], dim=-1).repeat_interleave(self.group_size, dim = -2).view(self.batch_size * added_length, num_heads, 2)
        v_param = torch.cat([v_scale, v_zero], dim=-1).repeat_interleave(self.group_size, dim = -2).view(self.batch_size * added_length, num_heads, 2)

        quantized_head_dim = self.pages.shape[-1]

        assert b_sz == self.batch_size
        if layer_idx == 0:
            current_length = self.length
            new_length = current_length + added_length
            self._ensure_page_cnt_per_batch(self.page_cnt_from_length(new_length))
            self.length = new_length
        attention_mask = cache_kwargs.get("attention_mask")
        if self._needs_init[layer_idx]:
            self._needs_init[layer_idx] = False
            if attention_mask is not None:
                nonzero_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten().view(-1, 1)
                key_states = key_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                value_states = value_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                key_states = torch.gather(key_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))
                value_states = torch.gather(value_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))

                k_param = k_param.view(self.batch_size * added_length, num_heads * 2)
                v_param = v_param.view(self.batch_size * added_length, num_heads * 2)
                k_param = torch.gather(k_param, 0, nonzero_indices.expand(-1, num_heads * 2))
                v_param = torch.gather(v_param, 0, nonzero_indices.expand(-1, num_heads * 2))

                seqlens_in_batch = torch.nn.functional.pad(torch.cumsum(attention_mask.sum(dim=-1, dtype=torch.int32), dim=0, dtype=torch.int32), (1, 0))
            else:
                seqlens_in_batch = torch.arange(self.batch_size + 1, device=self.device, dtype=torch.int) * added_length

            init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
            init_kv(
                **self.get_cache_specs_for_flash_infer(attention_mask),
                k=key_states.view(-1, num_heads, quantized_head_dim), 
                v=value_states.view(-1, num_heads, quantized_head_dim), 
                k_param=k_param.view(-1, num_heads, 2), 
                v_param=v_param.view(-1, num_heads, 2),
                seqlen_indptr=seqlens_in_batch,
                layer_idx=layer_idx
            )
            
            if key_states.dtype == torch.uint8 and self.trans.startswith("matmul"):
                return orig_key_states, orig_value_states # key_states_lac_only, value_states_lac_only # if we wants to use lac for prefill kv, use it
            else:
                return orig_key_states, orig_value_states
        else:
            assert added_length == 1
            append_kv = append_kv_f16 if self.disable_quant else append_kv_i4
            append_kv(
                **self.get_cache_specs_for_flash_infer(attention_mask),
                k=key_states.view(self.batch_size, num_heads, quantized_head_dim), 
                v=value_states.view(self.batch_size, num_heads, quantized_head_dim), 
                k_param=k_param.view(-1, num_heads, 2), 
                v_param=v_param.view(-1, num_heads, 2),
                layer_idx=layer_idx,
            )
        return functools.partial(
            self._stub.forward, 
            num_kv_heads=num_heads,
            attention_kwargs=self.get_cache_specs_for_flash_infer(attention_mask),
            layer_idx=layer_idx, 
        )
    
    def get_cache_specs_for_flash_infer(self, attention_mask):
        if attention_mask is not None:
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        else:
            seqlens_in_batch = torch.tensor([self.length], dtype=torch.int32, device=self.device).expand(self.batch_size)
        page_cnt = self.page_cnt_from_length(seqlens_in_batch)
        if (page_cnt[0] != page_cnt).any():
            raise NotImplementedError("Current implementation does not support the case where batches have different number of pages")
        page_cnt = page_cnt[0]
        page_ptr = seqlens_in_batch % self.page_size
        page_ptr = torch.where((seqlens_in_batch != 0) & (page_ptr == 0), self.page_size, page_ptr)
        return {
            f"kv_data": self.pages,
            f"kv_indptr": torch.arange(0, self.batch_size + 1, device=self.device, dtype=torch.int) * page_cnt, 
            f"kv_indices": (
                (torch.arange(page_cnt, device=self.device, dtype=torch.int) * self.batch_size).unsqueeze(0) + 
                torch.arange(self.batch_size, device=self.device, dtype=torch.int).unsqueeze(1)).view(-1), 
            f"last_page_offset": page_ptr, #torch.full((self.batch_size, ), page_ptr, device=self.device, dtype=torch.int),
            f"kv_param": self.scales, 
        }

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def to_legacy_cache(self):
        return self


class HFCacheAdapter(Cache):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_usable_length(self, kv_seq_len: int, layer_idx: int) -> int:
        return self.inner.get_usable_length(kv_seq_len, layer_idx)

    def get_seq_length(self) -> int:
        return self.inner.get_seq_length()

    def get_max_length(self) -> int:
        return self.inner.get_max_length()

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        return self.inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def to_legacy_cache(self):
        if hasattr(self.inner, "to_legacy_cache"):
            return self.inner.to_legacy_cache()
        return super().to_legacy_cache()