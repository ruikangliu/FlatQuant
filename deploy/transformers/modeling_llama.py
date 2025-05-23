import functools
import deploy
import deploy.transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, \
LlamaFlashAttention2, LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache


ALL_LAYERNORM_LAYERS.append(deploy.nn.RMSNorm)


class FlatQuantLlamaConfig(LlamaConfig):
    model_type = "llama_FlatQuant"


class FlatQuantFP16LlamaAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.inp_trans = torch.nn.Identity()
        self.o_proj_trans = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.inp_trans(hidden_states)
        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = self._flash_attention_forward(
                query_states, 
                key_states, 
                value_states, 
                query_length=q_len, 
                attention_mask=attention_mask
            )
        else:
            attn_output = cache_out(query_states)

        if isinstance(self.o_proj_trans, deploy.nn.OnlineTrans) and self.o_proj_trans.trans == "matmul":
            # attn_output: (bsz, seq_len, num_heads, head_dim)
            attn_output = self.o_proj_trans(attn_output.transpose(-1, -2).contiguous())
            attn_output.quantized_x = attn_output.quantized_x.transpose(-1, -2)
            attn_output.quantized_x = attn_output.quantized_x.reshape(bsz, q_len, -1).contiguous()
        else:
            attn_output = self.o_proj_trans(attn_output.transpose(-1, -2)).transpose(-1, -2)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FlatQuantLlamaAttention(FlatQuantFP16LlamaAttention):

    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options
        self.quantizer = deploy.nn.Quantizer()
        self.q_proj = deploy.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = deploy.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = deploy.nn.Linear4bit.from_float(self.v_proj)
        if "o_proj" in self.options.online_trans:
            self.o_proj_trans = deploy.nn.OnlineTrans(self.num_heads, trans=options.trans, decompose=False)
        self.o_proj = torch.nn.Sequential(
            deploy.nn.Quantizer(),
            deploy.nn.Linear4bit.from_float(self.o_proj)
        )
        if "qkv_proj" in self.options.online_trans:
            if not self.options.fuseLN:
                self.inp_trans = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)


class FlatQuantLlamaMLP(LlamaMLP):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options
        self.quantizer = deploy.nn.Quantizer()
        self.up_proj = deploy.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = deploy.nn.Linear4bit.from_float(self.gate_proj)
        if "down_proj" in self.options.online_trans:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.OnlineTrans(self.intermediate_size, trans=options.trans),
                deploy.nn.Quantizer(),
                deploy.nn.Linear4bit.from_float(self.down_proj)
            )
        else:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.Quantizer(),
                deploy.nn.Linear4bit.from_float(self.down_proj)
            )
        if "up_gate_proj" in self.options.online_trans:
            if not self.options.fuseLN:
                self.inp_trans = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)

    def forward(self, x):            
        if not self.options.fuseLN and hasattr(self, "inp_trans"):
            x = self.inp_trans(x)
        x = self.quantizer(x)
        return super().forward(x)


class FlatQuantFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, args=None):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FlatQuantFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None
        if args is not None:
            self.trans = args.trans
            self.online_trans = args.online_trans
        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_key_value_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16" 
        return deploy.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            trans_dtype=None if disable_quant else torch.float16,
            trans=self.trans if "qk" in self.online_trans else "none",
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=max_length,  # For now working with single page per batch.
                max_length=max_length)
        out = super().forward(input_ids, *args, past_key_values=past_key_values, **kwargs)
        return out
    

class FlatQuantLlamaForCausalLM(FlatQuantFP16LlamaForCausalLM):
    def __init__(self, args, config):
        super().__init__(config, args)
        assert config._attn_implementation == "flash_attention_2"
        if args.fuseLN:
            self.norm = deploy.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FlatQuantLlamaAttention(options=args, config=config, layer_idx=layer_idx)
            if args.fuseLN:
                layer.input_layernorm = deploy.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = deploy.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FlatQuantLlamaMLP(options=args, config=config)
        self.cache_dtype = "int4"
