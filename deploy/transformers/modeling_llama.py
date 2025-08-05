import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


import functools
import deploy
import deploy.transformers
from deploy.functional import get_decompose_dim
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, \
LlamaFlashAttention2, LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import Cache, DynamicCache


ALL_LAYERNORM_LAYERS.append(deploy.nn.RMSNorm)


class FlatQuantLlamaConfig(LlamaConfig):
    model_type = "llama_FlatQuant"


class FlatQuantFP16LlamaAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer_q = torch.nn.Identity()
        self.quantizer_k = torch.nn.Identity()
        self.quantizer_v = torch.nn.Identity()
        self.inp_trans_q = torch.nn.Identity()
        self.inp_trans_k = torch.nn.Identity()
        self.inp_trans_v = torch.nn.Identity()
        self.o_proj_trans = torch.nn.Identity()
        
        self._supports_cache_class = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_kwargs = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        self.isFlatQ = False
        if cache_kwargs is not None:
            self.isFlatQ = cache_kwargs.get("isFlatQ", False)

        if self.isFlatQ:
            #hidden_states = self.inp_trans(hidden_states)
            hidden_states_q = self.inp_trans_q(hidden_states)
            hidden_states_k = self.inp_trans_k(hidden_states)
            hidden_states_v = self.inp_trans_v(hidden_states)

            hidden_states_q = self.quantizer_q(hidden_states_q)
            hidden_states_k = self.quantizer_k(hidden_states_k)
            hidden_states_v = self.quantizer_v(hidden_states_v)
            
            query_states = self.q_proj(hidden_states_q)
            key_states = self.k_proj(hidden_states_k)
            value_states = self.v_proj(hidden_states_v)
        else:
            hidden_states = self.inp_trans_q(hidden_states)
            hidden_states = self.quantizer_q(hidden_states)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        group_size = self.num_heads // self.num_key_value_heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            # for llama2, 3
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            # for llama 3.1~
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_kwargs.update({
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
        })
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            trans_mat_for_q = cache_kwargs.get("trans_matrix_k_inv_t")
            if trans_mat_for_q is not None:
                query_states = torch.matmul(query_states.to(torch.float16), trans_mat_for_q) # trans for q TODO: fix hard-coded trans dtype
            attn_output = _flash_attention_forward(
                query_states, 
                key_states, 
                value_states, 
                query_length=q_len, 
                attention_mask=attention_mask,
                is_causal=True
            )
        else:
            attn_output = cache_out(query_states)

        if isinstance(self.o_proj_trans, deploy.nn.OnlineTrans) and self.o_proj_trans.trans == "matmul":
            # attn_output: (bsz, seq_len, num_heads, head_dim)
            attn_output = self.o_proj_trans(attn_output.transpose(-1, -2).contiguous())
            attn_output.quantized_x = attn_output.quantized_x.contiguous().reshape(bsz, q_len, -1)
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
        self.isFlatQ = hasattr(options, 'trans') and options.trans == "matmul" # if doing FlatQuant

        self.options = options
        self.quantizer_q = deploy.nn.Quantizer(lac = self.isFlatQ)
        self.quantizer_k = deploy.nn.Quantizer(lac = self.isFlatQ)
        self.quantizer_v = deploy.nn.Quantizer(lac = self.isFlatQ)
        self.q_proj = deploy.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = deploy.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = deploy.nn.Linear4bit.from_float(self.v_proj)
        if "o_proj" in self.options.online_trans:
            self.o_proj_trans = deploy.nn.OnlineTrans(self.num_heads, trans=options.trans, decompose=False)
        self.o_proj = torch.nn.Sequential(
            deploy.nn.Quantizer(lac = self.isFlatQ),
            deploy.nn.Linear4bit.from_float(self.o_proj)
        )
        if "qkv_proj" in self.options.online_trans:
            if not self.options.fuseLN:
                self.inp_trans_q = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
                self.inp_trans_k = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
                self.inp_trans_v = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
        
        num_heads = self.config.num_attention_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads

        self.register_buffer("trans_matrix_k", torch.randn([head_dim, head_dim], requires_grad = False))
        self.register_buffer("trans_matrix_k_inv_t", torch.randn([head_dim, head_dim], requires_grad = False))
        self.register_buffer("trans_matrix_v", torch.randn([head_dim, head_dim], requires_grad = False))
        self.register_buffer("kclip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("kclip_factor_a_min", torch.tensor(4.0))
        self.register_buffer("vclip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("vclip_factor_a_min", torch.tensor(4.0))


        left_dim, right_dim = get_decompose_dim(self.config.hidden_size)
        self.register_buffer("left_matrix", torch.randn([left_dim, left_dim], requires_grad = False))
        self.register_buffer("right_matrix", torch.randn([right_dim, right_dim], requires_grad = False))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        cache_kwargs = {
            "sin": kwargs.get("sin"),
            "cos": kwargs.get("cos"),
            "cache_position": cache_position,
            "attention_mask": attention_mask,
            "trans_matrix_k": self.trans_matrix_k,
            "trans_matrix_k_inv_t": self.trans_matrix_k_inv_t,
            "trans_matrix_v": self.trans_matrix_v,
            "kclip_factor_a_max": self.kclip_factor_a_max,
            "kclip_factor_a_min": self.kclip_factor_a_min,
            "vclip_factor_a_max": self.vclip_factor_a_max,
            "vclip_factor_a_min": self.vclip_factor_a_min,
            "isFlatQ": self.isFlatQ,
        }

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cache_kwargs=cache_kwargs,
        )

class FlatQuantLlamaMLP(LlamaMLP):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isFlatQ = hasattr(options, 'trans') and options.trans == "matmul" # if doing FlatQuant

        self.options = options
        self.quantizer_g = deploy.nn.Quantizer(lac = self.isFlatQ)
        self.quantizer_u = deploy.nn.Quantizer(lac = self.isFlatQ)
        self.up_proj = deploy.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = deploy.nn.Linear4bit.from_float(self.gate_proj)
        if "down_proj" in self.options.online_trans:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.OnlineTrans(self.intermediate_size, trans=options.trans),
                deploy.nn.Quantizer(lac = self.isFlatQ),
                deploy.nn.Linear4bit.from_float(self.down_proj)
            )
        else:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.Quantizer(lac = self.isFlatQ),
                deploy.nn.Linear4bit.from_float(self.down_proj)
            )
        if "up_gate_proj" in self.options.online_trans:
            if not self.options.fuseLN:
                self.inp_trans_g = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
                self.inp_trans_u = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
        
        left_dim, right_dim = get_decompose_dim(self.config.hidden_size)
        self.register_buffer("left_matrix", torch.randn([left_dim, left_dim], requires_grad = False))
        self.register_buffer("right_matrix", torch.randn([right_dim, right_dim], requires_grad = False))

    def forward(self, x):            
        if not self.options.fuseLN and hasattr(self, "inp_trans_g"): # isFlatQ
            x_up = self.up_proj(self.inp_trans_u(x))
            x_gate = self.gate_proj(self.inp_trans_g(x))
        else:
            x = self.quantizer_g(x)
            x_up = self.up_proj(x)
            x_gate = self.gate_proj(x)
        
        ac = self.act_fn(x_gate)
        x = x_up * ac
        x = self.down_proj(x)
        return x


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
        else:
            self.trans = "none"
            self.online_trans = set()
        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_attention_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // self.config.num_attention_heads
        group_size = num_heads // self.config.num_key_value_heads
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
            group_size = group_size
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None or isinstance(past_key_values, DynamicCache):
            max_length = max(self._expected_max_length or 0, input_ids.shape[1])
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=min(2048, max_length),  # For now working with single page per batch.
                max_length=max_length)
            past_key_values = deploy.transformers.HFCacheAdapter(past_key_values)
        elif not isinstance(past_key_values, Cache):
            past_key_values = deploy.transformers.HFCacheAdapter(past_key_values)

        kwargs.pop("use_cache", None)
        out = super().forward(input_ids, *args, past_key_values=past_key_values, use_cache = True, **kwargs)
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
        if hasattr(self, "generation_config"):
            self.generation_config.cache_implementation = None

        
    @classmethod
    def from_pretrained(cls, pretrained_model_name, **kwargs):
        import os
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from safetensors import safe_open
        import json
        from transformers.modeling_utils import no_init_weights

        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)

        config = cls.config_class.from_pretrained(pretrained_model_name, **kwargs)
        config._attn_implementation = "flash_attention_2"
        quant_config = getattr(config, 'quantization_config', {})

        class Args:
            pass

        args = Args()
        args.fuseLN = quant_config.get('fuseLN', False)
        args.trans = quant_config.get('trans', "matmul")
        args.online_trans = quant_config.get('online_trans', ["qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"])
        args.online_trans = set(args.online_trans)

        with no_init_weights():
            model = cls(args, config)

        state_dict = {}

        if os.path.isdir(pretrained_model_name):
            index_path = os.path.join(pretrained_model_name, "model.safetensors.index.json")
            single_path = os.path.join(pretrained_model_name, "model.safetensors")

            if os.path.exists(index_path):
                # Sharded model
                with open(index_path, 'r') as f:
                    index = json.load(f)
                
                loaded_files = set()
                for tensor_name, filename in index["weight_map"].items():
                    if filename not in loaded_files:
                        shard_path = os.path.join(pretrained_model_name, filename)
                        with safe_open(shard_path, framework = "pt") as f:
                            for key in f.keys():
                                if key in index["weight_map"] and index["weight_map"][key] == filename:
                                    state_dict[key] = f.get_tensor(key)
                        loaded_files.add(filename)
                        print(f"Loaded {filename}")
            else:
                # Single file
                state_dict = load_file(single_path)
        else:
            # Download from HuggingFace repo
            try:
                index_path = hf_hub_download(
                    repo_id=pretrained_model_name,
                    filename="model.safetensors.index.json"
                )
                
                # Sharded model - HF
                with open(index_path, 'r') as f:
                    index = json.load(f)
                
                loaded_files = set()
                for tensor_name, filename in index["weight_map"].items():
                    if filename not in loaded_files:
                        shard_path = hf_hub_download(
                            repo_id=pretrained_model_name,
                            filename=filename
                        )
                        with safe_open(shard_path, framework = "pt") as f:
                            for key in f.keys():
                                if key in index["weight_map"] and index["weight_map"][key] == filename:
                                    state_dict[key] = f.get_tensor(key)
                        loaded_files.add(filename)
                        print(f"Downloaded and loaded {filename}")
                        
            except Exception as e:
                # Single file
                try:
                    single_path = hf_hub_download(
                        repo_id=pretrained_model_name,
                        filename="model.safetensors",
                    )
                    state_dict = load_file(single_path)
                    print("Downloaded and loaded model.safetensors")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model weights: {e2}")
    
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
                        class Quantize:
                            pass
                        checkpoint["quantizers"][layer_name] = Quantize()
                    checkpoint["quantizers"][layer_name].scale = v
            else:
                checkpoint["model_state_dict"][k] = v

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

        model.load_state_dict(new_checkpoint["model_state_dict"], strict = False)
        model.load_state_dict(new_checkpoint["quantizers"], strict = False)

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

        torch.set_default_dtype(dtype_old)

        return model