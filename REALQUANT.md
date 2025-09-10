# Real Quantization for FlatQuant

Here, we add the explanation of the changes for real quantization of the fake quantized weights produced by FlatQuant algorithm, along with the pre-quantized weights in Huggingface and their performance.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    torch_device="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained("Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant")
streamer = TextStreamer(tokenizer)

prompt = "Summarize Barry Bonds's career so far as a legendary tale told by an old baseball coach.\n"

chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    chat, tokenize=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)


with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens = 200,
        do_sample = False,
        temperature = 1.0,
        streamer = streamer,
        pad_token_id = tokenizer.eos_token_id
    )
```
This might take a while to see the first token, as this compiles the FlatQuant kernel on the fly.

## Installation
1. Install the packages
    ```bash
    conda create -n flatquant python=3.10 -y
    conda activate flatquant
    # The requirements.txt is updated
    pip install -r requirements.txt
    pip install -e .
    pip install flash-attn --no-build-isolation
    ```

- To run models like LLaMA2, LLaMA3, we use `transformers==4.36.0` instead.

2. Download & link the models to `./modelzoo/` via running
    ```bash
    python get_snapshot_dir.py
    ```

- Unlike original FlatQuant repository, we use `./modelzoo/{model_type}/{hf_model_name}` format e.g. `./modelzoo/llama-3-instruct/llama-3-8b-instruct`.
- ⚠️ Be sure to use the correct **_CUDA.so** file that matches your environment and GPU. Using a .so file compiled in a different environment may lead to different kernel outputs. You can compile this file with `pip install -e .` in your environment.


## Usage

### Calibration

- Weight-Activation-KV Cache Quantization
    ```bash
    # W4A4KV4
    python ./main.py \
        --model ./modelzoo/llama-3/llama-3-8b \
        --w_bits 4 --a_bits 4 \
        --k_bits 4 --k_asym --k_groupsize 128 \
        --v_bits 4 --v_asym --v_groupsize 128 \
        --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
        --lwc --lac --cali_trans --add_diag \
        --output_dir ./outputs --save_matrix \
        --lm_eval --lm_eval_batch_size 16 \
        --quantized_save
    ```

### Evaluation & Check Speedup

```bash
python ./benchmarks/benchmark_model.py --batch_size 1
python ./benchmarks/benchmark_lm_eval.py --lm_eval_batch_size 16
```

- This code should be run with transformer==4.45.0
- If you want to check original speedup in the FlatQuant paper, use `--random_mode`. It is better to use transformer==4.36.0 with `git checkout bfd9e88` for results with high similarity to the original paper.
- ⚠️ Currently, only quantized models with WAKV sym_quantize are supported.

### Pre-quantized models in Huggingface

| Model                  |  URL                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLaMA-2-7B             |  [https://huggingface.co/Hyun9junn/Llama-2-7b-hf-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Llama-2-7b-hf-W4A4KV4-FlatQuant)                         |
| LLaMA-3-8B             |  [https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-W4A4KV4-FlatQuant)                     |
| LLaMA-3-8B-Instruct    |  [https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant)   |
| LLaMA-3-70B            |  [https://huggingface.co/Hyun9junn/Meta-Llama-3-70B-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Meta-Llama-3-70B-W4A4KV4-FlatQuant)                   |
| LLaMA-3.1-8B           |  [https://huggingface.co/Hyun9junn/Llama-3.1-8B-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Llama-3.1-8B-W4A4KV4-FlatQuant)                           |
| LLaMA-3.1-8B-Instruct  |  [https://huggingface.co/Hyun9junn/Llama-3.1-8B-Instruct-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Llama-3.1-8B-Instruct-W4A4KV4-FlatQuant)         |
| LLaMA-3.3-70B-Instruct |  [https://huggingface.co/Hyun9junn/Llama-3.3-70B-Instruct-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Llama-3.3-70B-Instruct-W4A4KV4-FlatQuant)       |


## Results

### Accuracy Results

**Table 1: WikiText-2 perplexity of 4-bit weight & acitvation quantized LLaMA models.**

| **Method**         | **W Quantizer** | **2-7B** | **3-8B**   | **3-70B** | **3-8B-Instuct** | **3.1-8B** | **3.1-8B-Instuct** | **3.1-70B-Instuct** | **3.3-70B-Instruct** |
| ------------------ | --------------- | -------- | ---------- | --------- | ---------------- | ---------- | ------------------ | ------------------- | -------------------- |
| FP16               | -               | 5.47     | 6.14       | 2.85      | 8.28             | 6.24       | 7.21               | 3.78                | 3.86                 |
| Fake-FlatQuant     | RTN             | 5.79     | 6.98       | 3.77      | 8.97             | 7.01       | 7.97               | 4.64                | 4.83                 |
| **Real-FlatQuant** | RTN             | **5.77** | **6.93**   | **3.74**  | **8.95**         | **6.95**   | **7.89**           | **4.60**            | **4.79**             |

**Table 2: Zero-shot QA task results of 4-bit weight & activation quantized LLaMA models.**

| **Method**         | **W Quantizer** | **2-7B**  | **3-8B**  | **3-70B** | **3-8B-Instuct** | **3.1-8B** | **3.1-8B-Instuct** | **3.1-70B-Instuct** | **3.3-70B-Instruct** |
| ------------------ | --------------- | --------- | --------- | --------- | ---------------- | ---------- | ------------------ | ------------------- | -------------------- |
| FP16               | -               | 69.81     | 73.26     | 80.03     | 72.54            | 74.04      | 73.76              | 78.41               | 78.39                |
| Fake-FlatQuant     | RTN             | 67.98     | 70.58     | 78.20     | 70.50            | 71.52      | **71.36**          | **77.69**           | 77.45                |
| **Real-FlatQuant** | RTN             | **68.11** | **71.18** | **78.51** | **70.83**        | **71.88**  | 71.26              | 77.45               | **77.65**            |


### Latency Results

**Table 3: Prefill speedup for batch sizes 1 on one RTX3090 GPU. We decode 256 tokens after the prefill on a sequence length of 2048.**

| **Model name**       | **Int4** | **QuaRot** | **FlatQuant** |
| -------------------- | -------- | ---------- | ------------- |
| LLaMA-2-7B           | 2.10     | 1.95       | 1.98          |
| LLaMA-3-8B           | 2.24     | 2.12       | 2.01          |
| LLaMA-3-8B-Instruct  | 2.24     | 2.12       | 2.06          |

**Table 4: Decoding speedup for batch sizes 1 on one RTX3090 GPU. We decode 256 tokens after the prefill on a sequence length of 2048.**

| **Model name**       | **Int4** | **QuaRot** | **FlatQuant** |
| -------------------- | -------- | ---------- | ------------- |
| LLaMA-2-7B           | 0.67     | 0.59       | 0.48          |
| LLaMA-3-8B           | 0.66     | 0.58       | 0.47          |
| LLaMA-3-8B-Instruct  | 0.67     | 0.58       | 0.48          |

## Changelog to enable real quantization
- Updated cutlass library version since the original version had the bug (see [here](https://github.com/ruikangliu/FlatQuant/issues/16)).
- Updated the requirements.txt to use later version of `torch` & transformer==4.45.0 for llama>3.1.
- Add `--quantized_save` flag to save real quantized weights from fake quantized weights.
- Add `benchmark_model.py` to measure **real speedup** for **whole model** with **real quantized weights**.
- Add `benchmark_lm_eval.py` to measure **zero-shot performance** for **real quantized model**.
- Upload real quantized model in Huggingface.
- Now this codebase can support llama2, llama3, llama3.1, llama3.2, llama3.3.
- Attention during the prefill uses unquantized query/keys as they are available during the prefill.
- Implemented **learned activation clipping** for FlatQuant kernels.
- ⚠️ Currently, only models with a `hidden_dim` that is a power of 2 are supported because of triton kernel implementation.
