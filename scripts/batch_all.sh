#!/bin/bash

SYSTEMS=b70

bash sbatch_model.sh ${SYSTEMS} -m  "facebook/MobileLLM-ParetoQ-1.5B-1.58-bit" -id Mobile-LLM-1.5B  -d logs -bs 1 -out 128 -in 32 "$@"
bash sbatch_model.sh ${SYSTEMS} -m "tiiuae/Falcon3-1B-Instruct" -id Falcon3-1B -d logs -bs 1 -out 128 -in 32 "$@"
bash sbatch_model.sh ${SYSTEMS} -m  "andrijdavid/Llama3-2B-Base" -id Llama3-2B -d logs -bs 1 -out 128 -in 32 "$@"
bash sbatch_model.sh ${SYSTEMS} -m meta-llama/Llama-3.1-8B -id Llama-3.1-8B  -d logs -bs 1 -out 128 -in 32 "$@"

