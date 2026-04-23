

# BASE_CMD="vllm serve --model ${MODEL} --max-model-len 2048 --max-num-seqs 64 --no-enable-prefix-caching --host 127.0.0.1 "
BASE_CMD="vllm bench latency --no-enable-prefix-caching --num-iters 10 --num-iters-warmup 3 --trust-remote-code --gpu-memory-utilization 0.9  "
# --model meta-llama/Llama-3.1-8B  --batch-size 1 --input 1024  --output-len 32 --max-model-len 2048

if python -c "import torch; print(torch.__version__);" |& grep "cu1" >& /dev/null ; then
	echo "Using CUDA"
elif python -c "import torch; print(torch.__version__);" |& grep "cpu" >& /dev/null ; then 
	echo "Using CPU"
	CORES_PER_SOCKET=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
	NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
	NUM_NUMA_NODES=`$PREFIX lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
	THREADS_PER_CORE=`$PREFIX lscpu | grep "Thread(s) per core:" | awk '{print $NF}'`
	CORES_PER_NUMA=$(( (CORES_PER_SOCKET * NUM_SOCKETS) / NUM_NUMA_NODES ))

	CORES=`lscpu | grep "NUMA node[0134] CPU" | awk '{print $NF}' | tr "," " "| awk -v ncpn=$CORES_PER_NUMA '{s=int($1);printf("%d-%d|",s, s+ncpn-1)}'`
	CORES=${CORES%?}
	echo ${CORES}

	export VLLM_CPU_KVCACHE_SPACE=40 
	export TORCHINDUCTOR_COMPILE_THREADS=1
	export VLLM_CPU_SGL_KERNEL=1
	export VLLM_CPU_OMP_THREADS_BIND="${CORES}"
elif python -c "import torch; print(torch.__version__);" |& grep "xpu" >& /dev/null ; then
	echo "Using XPU"
	#export TORCH_COMPILE_DISABLE=1
	# export SYCL_UR_USE_LEVEL_ZERO_V2=0
	export RenderCompressedBuffersEnabled=0
	export NEOReadDebugKeys=1
	export VLLM_XPU_ENABLE_XPU_GRAPH=1
	export ONEAPI_DEVICE_SELECTOR="opencl:1;level_zero:0"
	# export XETLA_QUANTIZE_LM_HEADS=1
else
	echo "Could not find torch version"
	if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
		exit 1
	else
		return
	fi
fi

PROFILE_CONFIG=""
for VAR in "$@" ; do
if [ ${VAR} == "--profile" ] ; then
PROFILE_CONFIG=$(cat <<EOF
'{"profiler": "torch", "torch_profiler_dir": "${PROF_DIR}", "torch_profiler_record_shapes": "True", "torch_profiler_with_stack": "False"}'
EOF
)
fi
done

if [ -z "${PROFILE_CONFIG}" ] ; then
echo "${BASE_CMD} " "$@"
eval "${BASE_CMD} " "$@"
else
echo "${BASE_CMD} " "$@" --profiler-config "${PROFILE_CONFIG}"
eval "${BASE_CMD} " "$@" --profiler-config "${PROFILE_CONFIG}"
fi


