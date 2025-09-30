from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

model_id_to_dir = {
    "meta-llama/Llama-2-7B-hf": "./modelzoo/llama-2/llama-2-7b-hf",
}

for model_id, local_dir in model_id_to_dir.items():

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    real_dir = snapshot_download(model_id, local_files_only=True)
    print("Model ID: ", model_id)
    print("Local directory: ", real_dir)
    print()

    subprocess.run(["ln", "-s", real_dir, local_dir])
