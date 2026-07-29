#!/usr/bin/env bash
# One-command launcher for the Bonsai int2 chat studio.
#
#   ./launch_demo.sh zen5              # allocate on partition "zen5" and serve
#   ./launch_demo.sh b70 --time 02:00:00
#   ./launch_demo.sh zen5 --jobid 346533        # reuse a specific allocation
#   ./launch_demo.sh --status                   # where is it running?
#   ./launch_demo.sh --stop                     # tear everything down
#
# The only required argument is the Slurm partition (the thing you would pass
# to `salloc --partition=...`). The script then:
#
#   1. reuses a RUNNING allocation on that partition, or creates one with
#      `salloc --no-shell`
#   2. resolves the compute node name
#   3. starts demo/server.py on that node (detached) and waits until the vLLM
#      engine reports ready
#   4. starts port_relay.py on the login node so VS Code / a browser can reach
#      it on localhost
#   5. prints the URL
#
# State lives in demo/.run/ so --status and --stop work from a new shell.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RUN_DIR="${SCRIPT_DIR}/.run"
STATE="${RUN_DIR}/state.env"
SERVER_LOG="${RUN_DIR}/server.log"
RELAY_LOG="${RUN_DIR}/relay.log"

TIME_LIMIT="03:59:00"
REMOTE_PORT="${REMOTE_PORT:-8000}"
LOCAL_PORT=""
PARTITION=""
JOBID=""
ACTION="start"
READY_TIMEOUT="${READY_TIMEOUT:-900}"     # seconds to wait for engine ready
EXTRA_SALLOC=()

msg()  { printf '\033[36m[demo]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[demo]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[demo]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    # Print the header comment block (everything from line 2 up to the first
    # non-comment line) as the help text.
    awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

# ---------------------------------------------------------------- arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)     TIME_LIMIT="$2"; shift 2 ;;
        --jobid)    JOBID="$2"; shift 2 ;;
        --port)     LOCAL_PORT="$2"; shift 2 ;;
        --remote-port) REMOTE_PORT="$2"; shift 2 ;;
        --stop)     ACTION="stop"; shift ;;
        --status)   ACTION="status"; shift ;;
        --logs)     ACTION="logs"; shift ;;
        -h|--help)  usage 0 ;;
        --)         shift; EXTRA_SALLOC+=("$@"); break ;;
        -*)         die "unknown option: $1 (see --help)" ;;
        *)          [[ -z "${PARTITION}" ]] || die "unexpected argument: $1"
                    PARTITION="$1"; shift ;;
    esac
done

mkdir -p "${RUN_DIR}"
# shellcheck disable=SC1090
[[ -f "${STATE}" ]] && source "${STATE}"

# ------------------------------------------------------------------ helpers
save_state() {
    cat > "${STATE}" <<EOF
JOB_ID="${JOB_ID:-}"
NODE="${NODE:-}"
REMOTE_PORT="${REMOTE_PORT:-}"
LOCAL_PORT="${LOCAL_PORT:-}"
RELAY_PID="${RELAY_PID:-}"
OWNED_ALLOCATION="${OWNED_ALLOCATION:-0}"
EOF
}

job_state() {   # $1 = jobid -> RUNNING / PENDING / "" if gone
    squeue -h -j "$1" -o "%T" 2>/dev/null | head -1
}

job_node() {    # $1 = jobid -> first node name
    squeue -h -j "$1" -o "%N" 2>/dev/null | head -1
}

free_port() {   # first free TCP port on the login node, starting at $1
    local p="$1"
    while [[ "${p}" -lt 65535 ]]; do
        if ! ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN; then
            echo "${p}"; return 0
        fi
        p=$((p + 1))
    done
    die "no free local port found"
}

server_alive() {
    [[ -n "${NODE:-}" ]] && \
    curl -s -m 5 --noproxy '*' -o /dev/null "http://${NODE}:${REMOTE_PORT}/health" 2>/dev/null
}

# A backend that is still loading the model does not answer /health yet, so
# also look for the process itself: starting a second engine on the same GPU
# makes both fight for memory and one of them dies.
# The bracket in '[u]vicorn' keeps the pattern from matching the command line
# of the wrapper shell that runs the pgrep (which would always report a hit).
backend_starting() {
    [[ -n "${JOB_ID:-}" ]] || return 1
    srun --jobid="${JOB_ID}" --overlap bash -lc \
        "pgrep -f '[u]vicorn server:app' >/dev/null" >/dev/null 2>&1
}

wait_for_ready() {
    # $1: 1 if we launched this backend (so SERVER_LOG describes it and its
    # failure marker is meaningful), 0 if we are attaching to someone else's.
    local own_log="${1:-0}"
    local deadline=$(( SECONDS + READY_TIMEOUT ))
    until server_alive; do
        # Only trust the log's failure marker for a backend we started, and
        # only once the process is really gone: the log can otherwise hold a
        # traceback from an earlier, unrelated attempt.
        if [[ "${own_log}" == "1" ]] \
           && grep -aq "Application startup failed" "${SERVER_LOG}" 2>/dev/null \
           && ! backend_starting; then
            tail -25 "${SERVER_LOG}" >&2
            die "backend failed to start (see ${SERVER_LOG})"
        fi
        if [[ "${own_log}" != "1" ]] && ! backend_starting; then
            die "the backend we were waiting for is gone; rerun to start a new one"
        fi
        (( SECONDS < deadline )) || die "backend not ready after ${READY_TIMEOUT}s (see ${SERVER_LOG})"
        sleep 5
    done
}

# --------------------------------------------------------------- stop/status
stop_all() {
    if [[ -n "${RELAY_PID:-}" ]] && kill -0 "${RELAY_PID}" 2>/dev/null; then
        kill "${RELAY_PID}" 2>/dev/null || true
        msg "relay stopped (pid ${RELAY_PID})"
    fi
    if [[ -n "${JOB_ID:-}" && -n "$(job_state "${JOB_ID}")" ]]; then
        # Kill the server step but keep the allocation unless we created it.
        # '[u]vicorn' so pkill does not match (and kill) this wrapper shell.
        srun --jobid="${JOB_ID}" --overlap bash -lc \
            "pkill -f '[u]vicorn server:app' || true" >/dev/null 2>&1 || true
        msg "backend stopped on ${NODE:-?} (job ${JOB_ID})"
        if [[ "${OWNED_ALLOCATION:-0}" == "1" ]]; then
            scancel "${JOB_ID}" 2>/dev/null || true
            msg "allocation ${JOB_ID} cancelled (it was created by this script)"
        else
            msg "allocation ${JOB_ID} left running (not created by this script)"
        fi
    fi
    rm -f "${STATE}"
}

print_status() {
    [[ -n "${JOB_ID:-}" ]] || { msg "nothing recorded in ${RUN_DIR}"; return 0; }
    local st; st="$(job_state "${JOB_ID}")"
    msg "job      : ${JOB_ID} (${st:-gone})"
    msg "node     : ${NODE:-?}:${REMOTE_PORT:-?}"
    if server_alive; then
        msg "backend  : ready"
    else
        msg "backend  : NOT responding"
    fi
    if [[ -n "${RELAY_PID:-}" ]] && kill -0 "${RELAY_PID}" 2>/dev/null; then
        msg "url      : http://localhost:${LOCAL_PORT}"
    else
        msg "relay    : not running"
    fi
    msg "logs     : ${SERVER_LOG}"
}

case "${ACTION}" in
    stop)   stop_all; exit 0 ;;
    status) print_status; exit 0 ;;
    logs)   exec tail -f "${SERVER_LOG}" ;;
esac

# ------------------------------------------------------------ 1. allocation
[[ -n "${PARTITION}" || -n "${JOBID}" ]] || usage 1

OWNED_ALLOCATION=0
if [[ -n "${JOBID}" ]]; then
    [[ -n "$(job_state "${JOBID}")" ]] || die "job ${JOBID} is not in the queue"
    JOB_ID="${JOBID}"
    msg "using allocation ${JOB_ID} (--jobid)"
else
    JOB_ID="$(squeue -h -u "${USER}" -p "${PARTITION}" -t RUNNING -o "%i" 2>/dev/null | head -1)"
    if [[ -n "${JOB_ID}" ]]; then
        msg "reusing RUNNING allocation ${JOB_ID} on partition ${PARTITION}"
    else
        msg "allocating a node on partition ${PARTITION} (--time ${TIME_LIMIT}) ..."
        # --no-shell: create the allocation and return, we drive it with srun.
        alloc_out="$(salloc --no-shell --partition="${PARTITION}" \
                            --time="${TIME_LIMIT}" \
                            ${EXTRA_SALLOC[@]+"${EXTRA_SALLOC[@]}"} 2>&1)" \
            || die "salloc failed:
${alloc_out}"
        JOB_ID="$(grep -oE 'job allocation [0-9]+' <<<"${alloc_out}" | grep -oE '[0-9]+' | head -1)"
        [[ -n "${JOB_ID}" ]] || die "could not parse a job id from salloc output:
${alloc_out}"
        OWNED_ALLOCATION=1
        msg "allocated job ${JOB_ID}"
    fi
fi

# Wait for the allocation to actually be RUNNING (it may start PENDING).
for _ in $(seq 1 120); do
    [[ "$(job_state "${JOB_ID}")" == "RUNNING" ]] && break
    sleep 5
done
[[ "$(job_state "${JOB_ID}")" == "RUNNING" ]] || die "job ${JOB_ID} did not start"

NODE="$(job_node "${JOB_ID}")"
[[ -n "${NODE}" ]] || die "could not resolve the node for job ${JOB_ID}"
msg "node     : ${NODE}"

# --------------------------------------------------------------- 2. backend
if server_alive; then
    msg "backend already serving on ${NODE}:${REMOTE_PORT}, reusing it"
elif backend_starting; then
    msg "a backend is already starting on ${NODE}, waiting for it "
    msg "(use --stop first if you want a fresh one)"
    wait_for_ready 0
    msg "backend ready"
else
    : > "${SERVER_LOG}"
    msg "starting backend on ${NODE}:${REMOTE_PORT} (log: ${SERVER_LOG})"
    setsid nohup srun --jobid="${JOB_ID}" --overlap bash -lc \
        "cd $(printf '%q' "${SCRIPT_DIR}") && PORT=${REMOTE_PORT} ./serve.sh" \
        > "${SERVER_LOG}" 2>&1 < /dev/null &
    disown

    msg "waiting for the engine (first start compiles the model, ~2 min) ..."
    wait_for_ready 1
    msg "backend ready"
fi

# ----------------------------------------------------------------- 3. relay
# The server listens on the compute node; VS Code port forwarding and browsers
# on the login node only see localhost, so bridge the two.
if [[ -n "${RELAY_PID:-}" ]] && kill -0 "${RELAY_PID}" 2>/dev/null; then
    kill "${RELAY_PID}" 2>/dev/null || true
fi
LOCAL_PORT="$(free_port "${LOCAL_PORT:-8765}")"
setsid nohup python "${SCRIPT_DIR}/port_relay.py" "${LOCAL_PORT}" "${NODE}" "${REMOTE_PORT}" \
    > "${RELAY_LOG}" 2>&1 < /dev/null &
RELAY_PID=$!
disown
sleep 1
kill -0 "${RELAY_PID}" 2>/dev/null || { cat "${RELAY_LOG}" >&2; die "relay failed to start"; }

save_state

# ---------------------------------------------------------------- 4. report
CFG_JSON="${RUN_DIR}/config.json"
if curl -s -m 10 --noproxy '*' "http://localhost:${LOCAL_PORT}/config" -o "${CFG_JSON}"; then
    python - "${CFG_JSON}" <<'PY' || true
import json, re, sys

with open(sys.argv[1]) as fh:
    cfg = json.load(fh)


def short_name(path):
    """HF cache dirs end in a commit hash; fall back to the repo name."""
    parts = [p for p in str(path or "?").split("/") if p]
    last = parts[-1] if parts else "?"
    if not re.fullmatch(r"[0-9a-f]{16,}", last, re.I):
        return last
    hub = next((p for p in parts if p.startswith("models--")), None)
    if hub:
        return hub[len("models--"):].split("--")[-1]
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i > 0:
            return parts[i - 1]
    return last


rows = [
    ("model", short_name(cfg.get("model"))),
    ("quantization", "{} / {}{}".format(
        cfg.get("quantization"), cfg.get("quant_method"),
        " (sidecar)" if cfg.get("prequantized") else "")),
    ("device", cfg.get("device")),
    ("context", cfg.get("max_model_len")),
    ("KV cache", "{:,} tokens".format(cfg.get("kv_cache_tokens") or 0)),
    ("images", "up to {}".format(cfg.get("max_images"))
     if cfg.get("supports_images") else "not supported"),
    ("load time", "{} s".format(cfg.get("load_seconds"))),
]
width = max(len(k) for k, _ in rows)
for key, value in rows:
    print("\033[36m[demo]\033[0m {:<{w}} : {}".format(key, value, w=width))
PY
fi

echo
msg "open  ->  http://localhost:${LOCAL_PORT}"
msg "logs  ->  ${BASH_SOURCE[0]##*/} --logs"
msg "stop  ->  ${BASH_SOURCE[0]##*/} --stop"
