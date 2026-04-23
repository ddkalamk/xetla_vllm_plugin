
if [ $# -gt 1 ] ; then
SYSTEMS="$1"
shift
else
	echo "Usage: $0 <comma separated systems names> <rest run_all_new.sh args>"
	exit 0
fi
if [ -z $SYSTEMS ] ; then
	echo "Please specify system name"
	exit 1
fi

echo "Running on ${SYSTEMS}"

bash ./run_all_new.sh -t xpu -s ${SYSTEMS} -tp 1 "$@"


