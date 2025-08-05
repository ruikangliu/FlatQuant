import json
import os
import shutil
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer

MAKE_MODE = False
TEST_MODE = False
UPLOAD_MODE = True

# LLaMA 2
LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
LLAMA2_70B = "meta-llama/Llama-2-70b-hf"

# LLaMA 3
LLAMA3_8B = "meta-llama/Meta-Llama-3-8B"
LLAMA3_70B = "meta-llama/Meta-Llama-3-70B"

LLAMA3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA31_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA31_8B = "meta-llama/Llama-3.1-8B"

model_name_col = [LLAMA31_8B]

for model_name in model_name_col:

    huggingface_name = os.path.basename(model_name)
    outputs_folder_name = huggingface_name.replace("Meta-", "").lower()
    huggingface_name = huggingface_name + "-W4A4KV4-FlatQuant"
    repo_name = f"Hyun9junn/{huggingface_name}"
    print(model_name)
    print(huggingface_name)
    print(outputs_folder_name)
    print(repo_name)

    save_dir = f"./final_model/{huggingface_name}"
    os.makedirs(save_dir, exist_ok = True)

    if MAKE_MODE:
        ## load config
        original_config = AutoConfig.from_pretrained(model_name)
        config_dict = original_config.to_dict()

        config_dict.update({
            "architectures": ["FlatQuantLlamaForCausalLM"],
            "auto_map": {
                "AutoModelForCausalLM": "modeling_llama.FlatQuantLlamaForCausalLM"
            },
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "flatquant",
                "w_bits": 4,
                "a_bits": 4,
                "kv_bits": 4,
                "w_is_symmetric": True,
                "a_is_symmetric": True,
                "kv_is_symmetric": False,
                "fuseLN": False,
                "trans": "matmul",
                "online_trans": ["qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"],
            }
        })

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent = 2)

        print("successfully loaded config!")

        ## load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)

        print("successfully loaded tokenizer!")

        ## copy deploy
        deploy_path = os.path.join(save_dir, "deploy")
        shutil.copytree("./deploy", deploy_path)

        print("successfully copied deploy!")

        ## copy quantized weights
        weight_dir = f"outputs/{outputs_folder_name}/w4a4/exp"

        if os.path.exists(os.path.join(weight_dir, "model.safetensors.index.json")):
            # Copy all shard files
            shard_files = glob.glob(os.path.join(weight_dir, "model-*.safetensors"))
            for shard_file in shard_files:
                shutil.copy2(shard_file, save_dir)
                print(f"Copied {os.path.basename(shard_file)}")
            
            # Copy index file
            shutil.copy2(os.path.join(weight_dir, "model.safetensors.index.json"), save_dir)
            print("Copied index file")
            
        else:
            # Single file
            shutil.copy2(os.path.join(weight_dir, "model.safetensors"), save_dir)
            print("Copied single safetensors file")

        print("successfully copied weights!")

        ## copy modeling_llama.py
        shutil.copy2("./deploy/transformers/modeling_llama.py", save_dir)

        print("successfully copied modelling_llama.py!")


        ## Generate README.md
        if "llama-3" in model_name.lower() or "meta-llama-3" in model_name.lower():
            license_name = "llama3"
        elif "llama-2" in model_name.lower():
            license_name = "llama2"
        else:
            license_name = "other"

        readme_content = f"""
---
license: {license_name}
base_model:
- {model_name}
---

# Model Card
- Base model: {model_name}
- Quantization method: FlatQuant

# How To Run
## Set Environment
```bash
git clone -b clean_squash --single-branch https://github.com/hyun9junn/FlatQuant.git FlatQuant

cd FlatQuant

conda create -n flatquant python=3.10 -y
conda activate flatquant
pip install -r requirements.txt
pip install -e .
pip install flash-attn --no-build-isolation
```

⚠️ **CUDA Required**: If you encounter CUDA-related errors, please check `nvcc --version` and install CUDA toolkit or set the path to `nvcc` correctly.

⚠️ Be sure to use the correct **_CUDA.so** file that matches your environment and GPU. Using a .so file compiled in a different environment may lead to different kernel outputs. You can compile this file with `pip install -e .` in your environment.

## Test Script
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_name}",  # Update this with your actual HF repo name
    trust_remote_code = True,
    torch_dtype = torch.float16,
    
)
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
streamer = TextStreamer(tokenizer)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model = model.to(device)

prompt = "Summarize Barry Bonds's career so far as a legendary tale told by an old baseball coach.\\n"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 50,
        do_sample = False,
        temperature = 1.0,
        streamer = streamer
    )
```

## Quantization Details
- Weight bits: {config_dict['quantization_config']['w_bits']}
- Activation bits: {config_dict['quantization_config']['a_bits']}
- KV cache bits: {config_dict['quantization_config']['kv_bits']}
- Weight symmetric: {config_dict['quantization_config']['w_is_symmetric']}
- Activation symmetric: {config_dict['quantization_config']['a_is_symmetric']}
- KV cache symmetric: {config_dict['quantization_config']['kv_is_symmetric']}
"""

        readme_path = os.path.join(save_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print("successfully generated README.md!")


    if TEST_MODE:
        ## test
        print("test at local..")

        model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code = True,
            torch_dtype=torch.float16,
            
        )

        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        streamer = TextStreamer(tokenizer)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)

        prompt = "Summarize Barry Bonds's career so far as a legendary tale told by an old baseball coach.\n"

        inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
        print(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 충분한 길이
                do_sample=False,
                streamer=streamer
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens = True)


    if UPLOAD_MODE:
        from huggingface_hub import HfApi
        api = HfApi()

        # make new repo
        api.create_repo(
            repo_name,
            repo_type = "model",
            exist_ok = True,
            private = False,
        )
        print("successfully made new repo")


        # upload folder
        api.upload_folder(
            folder_path = save_dir,
            repo_id = repo_name,
            repo_type = "model",
            commit_message = "Initial upload"
        )

        print("Upload completed!")