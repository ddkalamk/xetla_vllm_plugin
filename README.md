# One time Setup
```bash
# Setup vLLM Python uv ENV
bash utils/setup_vllm_xpu.sh
```

# Install oneAPI Deep Learning Essentials
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/56f7923a-adb8-43f3-8b02-2b60fcac8cab/intel-deep-learning-essentials-2025.3.3.16_offline.sh
bash ./intel-deep-learning-essentials-2025.3.3.16_offline.sh -a --silent --eula accept
```


# Running latency benchmark
```bash
# activate uv env
source .venv/bin/activate

MODEL_NAME="Qwen/Qwen2.5-1.5B"
# example bf16 command line
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048

# use fp8 quantized gemms
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048 -q fp8

# use xetla int2-bf16 gemms
vllm bench latency --model "$MODEL_NAME"  --batch-size 1 --num-iters 10 --num-iters-warmup 3 --gpu_memory_util=0.7 --input-len 32 --output-len 128 --max-model-len 2048 -q xetla

