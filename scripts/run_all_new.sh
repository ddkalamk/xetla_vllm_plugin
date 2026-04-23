#!/bin/bash

# Initialize variables with default values
MODEL="facebook/opt-125m"
MODELID=""
SYSTAG=test
TYPE=""
TP=1
LOG_DIR=logs
EAGER=2
NO_PROFILE=0
ONLY_PROFILE=0


BATCHES="1,16"
INPUTS="1024"
OUTPUTS="1,32"
NO_PROFILE=0
ONLY_PROFILE=0
CPU_MAX_BATCH=${CPU_MAX_BATCH:=32}

function csv2list {
	echo $1 | tr "," " "
}

# Loop through all arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2 # Shift past the flag and the value
      ;;
    -s|--sys-tag)
      SYSTAG="$2"
      shift 2
      ;;
    -t|--type)
      TYPE="$2"
      shift 2
      ;;
    -d|--log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -tp|--tensor-parallel)
      TP="$2"
      shift 2
      ;;
    -id|--model-id)
      MODELID="$2"
      shift 2
      ;;
    -in|--inputs)
      INPUTS="$2"
      shift 2
      ;;
    -out|--outputs)
      OUTPUTS="$2"
      shift 2
      ;;
    -bs|--batch-sizes)
      BATCHES="$2"
      shift 2
      ;;
    -eager|--enforce-eager)
      EAGER=1
      shift 1
      ;;
    --skip-profile)
      NO_PROFILE=1
      shift 1
      ;;
    --only-profile)
      ONLY_PROFILE=1
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  -id, --model-id <name>  Set model tag name for log file"
      echo "  -m, --model <name>  Set model name"
      echo "  -stg, --sys-tag <name> Set SYSTAG for logfile name (e.g. emr1s g31)"
      echo "  -t, --type <cpu|xpu|cuda> "
      echo "  -tp, --tensor-parallel <num_tp_ranks> "
      echo "  -d, --log-dir <log_dir_name (logs)> "
      echo "  -in, --inputs <comma separated input prompt sizes> "
      echo "  -out, --outputs <comma separated num output tokens "
      echo "  -bs, --batch-sizes <comma separated batch sizes "
      echo "  -eager, --enforce-eager "
      echo "  --skip-profile "
      echo "  --only-profile "
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


if [ -z "${MODELID}" ] ; then
	MODELID=`echo $MODEL | tr "/" " " | awk '{print $NF}'`
fi

if [ -z "${TYPE}" ] ; then
	echo "Must specify system type using -t [cpu|cuda|xpu]"
	exit 1
fi

TYPE_IS_VALID=0
for VALID_TYPE in cpu cuda xpu ; do
	if [ $TYPE == $VALID_TYPE ] ; then TYPE_IS_VALID=1; break; fi
done
if [ $TYPE_IS_VALID -eq 0 ] ; then
	echo "Must specify system type using -t [cpu|cuda|xpu]"
	echo "Current TYPE=${TYPE}"
	exit 1
fi
TAG="${SYSTAG}_${MODELID}"
BATCHES=( $(csv2list ${BATCHES}) )
INPUTS=( $(csv2list ${INPUTS}) )
OUTPUTS=( $(csv2list ${OUTPUTS}) )
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS}

echo "Batches: ${BATCHES[@]}"
echo "Inputs: ${INPUTS[@]}"
echo "Outputs: ${OUTPUTS[@]}"
echo "Type: $TYPE"

HST=`hostname -s`

source env.sh $TYPE
BASE_LOG_DIR=${LOG_DIR}
# LOG_DIR=${LOG_DIR:=logs}
for PREC in bf16 int2 ; do
LOG_DIR=${BASE_LOG_DIR}_${PREC}
mkdir -p ${LOG_DIR}
if [ $PREC == "int2" ] ; then
	EXTRA_VLLM_ARGS=" -q xetla "
else
	EXTRA_VLLM_ARGS=""
fi

if [ ${EAGER} -eq 1 ] ; then
	EXTRA_VLLM_ARGS=" --enforce-eager ${EXTRA_VLLM_ARGS}"
fi

for BS in ${BATCHES[@]} ; do
for INP in ${INPUTS[@]} ; do
for OUT in ${OUTPUTS[@]} ; do
	if [ $TYPE == "cpu" ] && [ $BS -gt ${CPU_MAX_BATCH} ] ; then continue ; fi
	MAX_MODEL_LEN=$(( INP + OUT + 100 ))
	LOG_KEY=${TAG}_tp_${TP}_bs_${BS}_i_${INP}_o_${OUT}_eager_${EAGER}_${HST}
	export PROF_DIR=${LOG_DIR}/prof_${LOG_KEY}_trace
	LOG_PROF_FILE=${LOG_DIR}/prof_${LOG_KEY}.txt
	LOG_FILE=${LOG_DIR}/out_${LOG_KEY}.txt
	
	# if test -f $LOG_FILE && grep "Avg latency" $LOG_FILE >& /dev/null ; then 
	# 	echo "Valid $LOG_FILE exists, skipping..."
	# 	continue
	# fi
	# echo $LOG_FILE
	echo "bash run_vllm_latency_bench.sh --model $MODEL --batch-size $BS --input-len $INP  --output-len $OUT --max-model-len $MAX_MODEL_LEN -tp $TP ${EXTRA_VLLM_ARGS} |& tee -a $LOG_FILE"
	CMD="bash run_vllm_latency_bench.sh --model $MODEL --batch-size $BS --input-len $INP  --output-len $OUT --max-model-len $MAX_MODEL_LEN -tp $TP ${EXTRA_VLLM_ARGS}"
	if [ ${ONLY_PROFILE} -eq 0 ] ; then
	echo $CMD |& tee $LOG_FILE
	eval $CMD |& tee -a $LOG_FILE
	fi
	if [ $NO_PROFILE -eq 0 ] ; then
	CMD="${CMD} --profile"
	echo $CMD |& tee $LOG_PROF_FILE
	eval $CMD |& tee -a $LOG_PROF_FILE
	fi
done
done
done
done

