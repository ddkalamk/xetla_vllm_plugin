
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "This script must be sourced, exiting..."
fi

if [ "x$1" == "x" ] ; then
	echo "Please specify xpu or cpu or cuda as first argument"
	echo "Use: source ${BASH_SOURCE[0]} xpu|cpu|cuda"
	return 1
elif [ $1 == "xpu" ] ; then
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	# export SYCL_UR_USE_LEVEL_ZERO_V2=0
	source ${SCRIPT_DIR}/../.venv/bin/activate
else
	echo "Unsupported backend \"$1\""
fi

