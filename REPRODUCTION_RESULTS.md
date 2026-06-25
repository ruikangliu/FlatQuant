# FlatQuant Reproduction Results

## Setup
- Hardware: 1× NVIDIA RTX 4090 (24 GB) used for the run (4 available)
- Env: uv-managed Python 3.10.20 (venv at `source/.venv`); torch 2.6.0+cu124, transformers 4.45.0, lm-eval 0.4.9.1
- fast_hadamard_transform: built from source (Dao-AILab GitHub); PyPI sdist is broken (missing csrc)
- Model: Qwen2.5-7B-Instruct (open weights) — local dir ./modelzoo/qwen-2.5-7b-instruct
- Calibration set: WikiText-2 (train); Eval: WikiText-2 (test) + C4 (validation)

## Config (W4A4KV4, RTN weight quantizer)
Matches scripts/qwen-2.5-instruct/qwen-2.5-instruct-7b/w4a4kv4.sh except cali_bsz:
- w_bits 4, a_bits 4, k_bits 4 (asym, g128), v_bits 4 (asym, g128)
- nsamples 128, epochs 15, flat_lr 5e-3, lwc + lac + cali_trans + add_diag
- deactive_amp (fp32), direct_inv
- cali_bsz 2  (paper uses 4; reduced to fit 24 GB — bsz=4 OOMs at ~22.4 GB.
  Same nsamples/epochs/lr; scheduler step count adapts via nsamples//cali_bsz.)

## Results (WikiText-2 / C4 perplexity, seqlen 2048)

| Source                 | WikiText-2 | C4    |
| ---------------------- | ---------- | ----- |
| Paper Table 4 (FlatQuant RTN W4A4) | 8.46 | 13.94 |
| **This reproduction**  | **8.29**   | **13.84** |
| Paper BF16 baseline    | 8.36       | 14.37 |

Reproduced numbers match the paper closely (both within ~0.2 PPL, and slightly
better than the reported values). Reproduction confirmed.

## Runtime
- Calibration: ~2h31m (28 layers, ~5.4 min/layer on one 4090)
- RTN quant + PPL eval: ~5 min

## How to run (reproduce)

```bash
cd /home/maomaotat/xwh/FlatQuant/source
source .venv/bin/activate          # venv lives here, under source/
export CUDA_HOME=/home/maomaotat/.local/cuda PATH=$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ./main.py \
    --model ./modelzoo/qwen-2.5-7b-instruct \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 2 --nsamples 128 --epochs 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --exp_name w4a4kv4_repro --save_matrix \
    --deactive_amp --direct_inv
```

To re-evaluate without re-calibrating, add `--reload_matrix --matrix_path ./outputs/qwen-2.5-7b-instruct/w4a4/w4a4kv4_repro`.
