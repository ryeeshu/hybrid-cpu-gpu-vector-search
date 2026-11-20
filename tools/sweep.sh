#!/usr/bin/env bash
# Measurement sweep for hvs_search: timed rounds, verification runs, transfer
# benchmark and visit counts, written to one new results directory. With
# --extend it adds one more timed round of some configurations to an existing
# results directory instead.
# Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
#
# HVS_REPO overrides the repository root (default: the parent of this
# script's directory); it is used only to ask git about uncommitted changes.
set -euo pipefail
export LC_ALL=C

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=${HVS_REPO:-$(dirname "$here")}
orig_args=("$@")

build=""
data=""
out=""
label=""
query_count=""
extend=""
only=""
reason=""
rounds=3
rounds_given=0
pause_s=30
skip_xfer=0
skip_visits=0
dry_run=0

# The ef values of every timed launch, and timed repetitions per row.
ef_values=(16 32 64 128 256)
repetitions=3

# Timed configurations: name and hvs_search arguments, in rotation order.
config_names=(v1_t1 v1_t8 v2_t1 v2_t8 v2stage_t1 v2stage_t8 v2cpu_t1 v2cpu_t8)
config_args=(
    "--mode cpu --threads 1"
    "--mode cpu --threads 8"
    "--mode hybrid-sync --backend cuda --threads 1"
    "--mode hybrid-sync --backend cuda --threads 8"
    "--mode hybrid-sync --backend cuda --stage-timing --threads 1"
    "--mode hybrid-sync --backend cuda --stage-timing --threads 8"
    "--mode hybrid-sync --backend cpu --threads 1"
    "--mode hybrid-sync --backend cpu --threads 8"
)
# With --query-count, V1 on the full query set joins the rotation as well.
v1full_names=(v1full_t1 v1full_t8)
v1full_args=("--mode cpu --threads 1" "--mode cpu --threads 8")

die() {
    echo "sweep: $*" >&2
    exit 1
}

usage() {
    cat >&2 << 'EOF'
usage: tools/sweep.sh --build DIR --data DIR --out DIR --label TEXT
           [--query-count N] [--rounds 3] [--pause 30] [--skip-xfer]
           [--skip-visits] [--dry-run]
       tools/sweep.sh --build DIR --data DIR --extend RESULTS_DIR
           --only NAME[,NAME...] --reason TEXT [--query-count N] [--pause 30]
           [--dry-run]
EOF
    exit 2
}

is_uint() {
    [[ $1 =~ ^[0-9]+$ ]]
}

# The words as one line, quoted only where the shell would need it.
quote() {
    local word out=""
    for word in "$@"; do
        if [[ $word =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then
            out+="$word "
        else
            out+="$(printf '%q' "$word") "
        fi
    done
    printf '%s' "${out% }"
}

# Runs a command, or only prints it with --dry-run.
step() {
    if (( dry_run )); then
        echo "+ $(quote "$@")"
    else
        "$@"
    fi
}

# Number of rows in the header of an .fbin file.
fbin_rows() {
    od -An -tu4 -N4 "$1" | tr -d ' \n'
}

while (( $# )); do
    case $1 in
        --build | --data | --out | --label | --query-count | --rounds | --pause \
            | --extend | --only | --reason)
            (( $# >= 2 )) || die "$1 needs a value"
            case $1 in
                --build) build=$2 ;;
                --data) data=$2 ;;
                --out) out=$2 ;;
                --label) label=$2 ;;
                --query-count) query_count=$2 ;;
                --rounds) rounds=$2; rounds_given=1 ;;
                --pause) pause_s=$2 ;;
                --extend) extend=$2 ;;
                --only) only=$2 ;;
                --reason) reason=$2 ;;
            esac
            shift 2
            ;;
        --skip-xfer) skip_xfer=1; shift ;;
        --skip-visits) skip_visits=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h | --help) usage ;;
        *) die "unknown option $1" ;;
    esac
done

if [[ -n $extend ]]; then
    [[ -n $build && -n $data ]] || usage
    [[ -z $out && -z $label ]] ||
        die "--extend takes the directory and label from the results, not --out or --label"
    (( ! rounds_given )) || die "--rounds does not apply with --extend"
    [[ -n $only ]] || die "--extend needs --only"
    # The reason is written into launches.log, a tab separated file.
    reason=${reason//[$'\t\n\r']/ }
    [[ -n ${reason//[[:space:]]/} ]] || die "--extend needs a --reason"
    out=$extend
    skip_xfer=1
    skip_visits=1
else
    [[ -n $build && -n $data && -n $out && -n $label ]] || usage
    [[ -z $only && -z $reason ]] || die "--only and --reason need --extend"
fi
is_uint "$rounds" && (( rounds >= 1 )) || die "--rounds needs a positive number"
is_uint "$pause_s" || die "--pause needs a number of seconds"
if [[ -n $query_count ]]; then
    is_uint "$query_count" && (( query_count >= 1 )) ||
        die "--query-count needs a positive number"
fi
# The label is written into the CSV unquoted.
case $label in
    *[,\"\']* | *$'\n'* | *$'\r'*) die "--label must not contain commas, quotes or newlines" ;;
esac
(( dry_run )) || [[ -d $data ]] || die "no data directory $data"
(( dry_run )) || [[ -d $build ]] || die "no build directory $build"

# Results are never overwritten; --extend only appends rows and log lines.
if [[ -n $extend ]]; then
    for name in search.csv launches.log build_manifest.txt; do
        [[ -r $out/$name ]] || die "$out has no $name; --extend needs the results of a sweep"
    done
else
    [[ ! -e $out ]] || die "$out already exists; choose a new --out"
fi

# Query and learn row counts come from the .fbin headers.
full_queries=0
learn_rows=0
needed=(base.fbin query.fbin gt.ibin graph0.bin upper.bin)
(( skip_visits )) || needed+=(learn.fbin)
for name in "${needed[@]}"; do
    [[ -r $data/$name ]] || (( dry_run )) || die "cannot read $data/$name"
done
if [[ -r $data/query.fbin ]]; then
    full_queries=$(fbin_rows "$data/query.fbin")
fi
if [[ -r $data/learn.fbin ]]; then
    learn_rows=$(fbin_rows "$data/learn.fbin")
fi
if [[ -z $extend && -n $query_count ]] &&
    (( full_queries > 0 && query_count >= full_queries )); then
    die "--query-count $query_count is not below the $full_queries queries in $data/query.fbin"
fi

# What the existing rows of --extend say, on four lines: the build id, the
# tag, the --query-count of the sweep ("" if none) and the largest launch.
# Arguments: RESULTS_DIR ONLY QUERY_COUNT, then each configuration name
# followed by its hvs_search arguments. Exits with a message if --only names
# an unknown configuration or one without rows, or if QUERY_COUNT differs
# from the queries of those rows.
describe_results() {
    python3 - "$@" << 'EOF'
import csv
import re
import sys

results, only, wanted = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
known = dict(zip(sys.argv[4::2], sys.argv[5::2]))


def fail(text):
    sys.exit("sweep: " + text)


def query_count_of(row):
    found = re.search(r" --query-count ([0-9]+)", row["command"])
    return found.group(1) if found else ""


with open(results + "/search.csv", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    fail("%s/search.csv has no rows" % results)
for column in ("build_id", "tag"):
    values = sorted({row[column] for row in rows})
    if len(values) != 1:
        fail("%s/search.csv has %d values of %s (%s), not one" % (
            results, len(values), column, ", ".join(values)))
counts = {query_count_of(row) for row in rows} - {""}
if len(counts) > 1:
    fail("the launches in %s/search.csv used --query-count %s" % (
        results, " and ".join(sorted(counts))))
sweep_count = counts.pop() if counts else ""
if len(set(only)) < len(only):
    fail("--only names a configuration twice")
queries = set()
for name in only:
    if name not in known:
        fail("unknown configuration '%s'; the names are %s" % (
            name, " ".join(known)))
    words = known[name].split()
    mode = words[words.index("--mode") + 1]
    backend = words[words.index("--backend") + 1] if "--backend" in words else ""
    stage = "" if mode == "cpu" else str(int("--stage-timing" in words))
    threads = words[words.index("--threads") + 1]
    full = name.startswith("v1full")
    if full and not sweep_count:
        fail("%s has no rows: the sweep in %s used no --query-count" % (
            name, results))
    # V1 and V1 on the full queries differ only in the --query-count.
    mine = [row for row in rows
            if (row["mode"], row["backend"], row["stage_timing"],
                row["threads"]) == (mode, backend, stage, threads)
            and (not sweep_count or (query_count_of(row) == "") == full)]
    if not mine:
        fail("%s/search.csv has no rows of %s" % (results, name))
    if not full:
        queries |= {row["queries"] for row in mine}
if len(queries) > 1:
    fail("the rows of %s have %d different query counts" % (
        " ".join(only), len(queries)))
used = queries.pop() if queries else sweep_count
if wanted and wanted != used:
    fail("--query-count %s does not match the %s queries of the existing "
         "rows of %s" % (wanted, used, " ".join(only)))
print(rows[0]["build_id"])
print(rows[0]["tag"])
print(sweep_count)
print(max(int(row["launch"]) for row in rows))
EOF
}

existing_id=""
extra_launch=0
if [[ -n $extend ]]; then
    if grep -q $'^extend\t' "$out/launches.log"; then
        die "$out already has an extra round; the rule allows one"
    fi
    pairs=()
    for (( i = 0; i < ${#config_names[@]}; i++ )); do
        pairs+=("${config_names[i]}" "${config_args[i]}")
    done
    for (( i = 0; i < ${#v1full_names[@]}; i++ )); do
        pairs+=("${v1full_names[i]}" "${v1full_args[i]}")
    done
    existing=$(describe_results "$out" "$only" "$query_count" "${pairs[@]}") || exit 1
    { read -r existing_id; read -r label; read -r query_count; read -r rounds; } <<< "$existing"
    extra_launch=$(( rounds + 1 ))
fi

hvs=$build/hvs_search
xfer=$build/xfer_bench
tmp=""
cleanup() {
    if [[ -n $tmp ]]; then
        rm -rf "$tmp"
    fi
    # Removes the results directory only if nothing was written to it.
    rmdir "$out" 2> /dev/null || true
}
trap cleanup EXIT

# 1. The results directory.
if (( ! dry_run )) && [[ -z $extend ]]; then
    mkdir -p "$(dirname "$out")"
    mkdir "$out"
fi

# 2. Thread placement is the same for every launch.
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
unset OMP_NUM_THREADS OMP_THREAD_LIMIT

# 3. Build and test, then the build id.
step cmake --build "$build" || die "the build failed"
step ctest --test-dir "$build" --output-on-failure || die "the tests failed"
if (( dry_run )); then
    id=${existing_id:-dry-run}
    echo "+ $hvs --build-id${existing_id:+   (must print $existing_id)}"
else
    [[ -x $hvs ]] || die "$hvs is not built"
    id=$("$hvs" --build-id)
    [[ $id =~ ^[0-9a-f]{12}$ ]] || die "build id '$id' is not 12 hex digits"
    [[ -r $build/build_manifest.txt ]] || die "no $build/build_manifest.txt"
    if [[ -n $extend && $id != "$existing_id" ]]; then
        die "build id mismatch: $hvs prints $id, $out/search.csv has $existing_id; never mix builds"
    fi
fi
if [[ -z $extend ]]; then
    step cp "$build/build_manifest.txt" "$out/build_manifest.txt"
fi

# 4. The conditions of the sweep.
first_line() {
    # The first line of stdin, or na.
    local line=""
    IFS= read -r line || true
    echo "${line:-na}"
}

read_or_na() {
    if [[ -r $1 ]]; then first_line < "$1"; else echo na; fi
}

gpu_query() {
    # nvidia-smi --query-gpu fields, one GPU per ';', or na.
    if command -v nvidia-smi > /dev/null 2>&1; then
        { nvidia-smi --query-gpu="$1" --format=csv,noheader 2> /dev/null || true; } |
            paste -sd ';' | first_line
    else
        echo na
    fi
}

mean_mhz() {
    awk '/^cpu MHz/ { s += $4; n++ } END { if (n) printf "%.0f\n", s / n; else print "na" }' \
        /proc/cpuinfo 2> /dev/null || echo na
}

cpu_temp() {
    # Package temperature in degrees C, or na.
    local zone hwmon milli
    for zone in /sys/class/thermal/thermal_zone*; do
        [[ -r $zone/type && -r $zone/temp ]] || continue
        if [[ $(first_line < "$zone/type") == x86_pkg_temp ]]; then
            read -r milli < "$zone/temp" 2> /dev/null || continue
            echo $(( milli / 1000 ))
            return
        fi
    done
    for hwmon in /sys/class/hwmon/hwmon*; do
        [[ -r $hwmon/name && -r $hwmon/temp1_input ]] || continue
        case $(first_line < "$hwmon/name") in
            coretemp | k10temp)
                read -r milli < "$hwmon/temp1_input" 2> /dev/null || continue
                echo $(( milli / 1000 ))
                return
                ;;
        esac
    done
    echo na
}

logins() {
    who | wc -l
}

uncommitted() {
    # yes, no or unknown; needs a .git directory and git.
    local status
    if [[ -d $repo/.git ]] && command -v git > /dev/null 2>&1; then
        if status=$(git -C "$repo" status --porcelain 2> /dev/null); then
            if [[ -n $status ]]; then echo yes; else echo no; fi
            return
        fi
    fi
    echo unknown
}

write_env() {
    local cpu_model nvcc_line compiler_line
    cpu_model=$(grep -m1 'model name' /proc/cpuinfo 2> /dev/null | cut -d: -f2- |
        sed 's/^ //' | first_line)
    if command -v nvcc > /dev/null 2>&1; then
        nvcc_line=$(nvcc --version 2> /dev/null | tail -n 1)
    else
        nvcc_line=na
    fi
    compiler_line=$(grep -i -m1 compiler "$build/build_manifest.txt" 2> /dev/null | first_line)
    {
        echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "hostname: $(uname -n)"
        echo "kernel: $(uname -a)"
        echo "cpu_model: $cpu_model"
        echo "logical_cpus: $(nproc --all)"
        echo "cpu_governor: $(read_or_na /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
        echo "transparent_hugepage: $(read_or_na /sys/kernel/mm/transparent_hugepage/enabled)"
        echo "gpu_name: $(gpu_query name)"
        echo "gpu_driver: $(gpu_query driver_version)"
        echo "gpu_memory: $(gpu_query memory.total)"
        echo "nvcc: $nvcc_line"
        echo "compiler (manifest): $compiler_line"
        echo "OMP_PROC_BIND: ${OMP_PROC_BIND-unset}"
        echo "OMP_PLACES: ${OMP_PLACES-unset}"
        echo "OMP_DYNAMIC: ${OMP_DYNAMIC-unset}"
        echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS-unset}"
        echo "OMP_THREAD_LIMIT: ${OMP_THREAD_LIMIT-unset}"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-unset}"
        echo "uptime: $(uptime)"
        echo "logins: $(logins)"
        echo "sweep command: $(quote "$0" "${orig_args[@]}")"
        echo "build id: $id"
        echo "source tree uncommitted: $(uncommitted)"
    } > "$out/env.txt"
}

log=$out/launches.log
console=$out/console.log
if [[ -n $extend ]]; then
    : # env.txt and the header of launches.log are those of the sweep
elif (( dry_run )); then
    echo "+ write $out/env.txt"
else
    write_env
    echo "# tab separated; start: time, round, name, command, load1, logins, mhz, temp_c, gpu" > "$log"
    echo "# end: time, round, name, exit, elapsed_s" >> "$log"
    echo "# check: time, exit, first line of check_results.py output" >> "$log"
    echo "# final_build_id: time, build id after the sweep, unchanged or changed" >> "$log"
fi

# One launch: pause, log the conditions, run, log the exit code and time.
#   launch ROUND NAME PAUSE_SECONDS STDOUT_FILE COMMAND...
# An empty STDOUT_FILE leaves the output on the terminal.
launch() {
    local round=$1 name=$2 wait_s=$3 stdout_file=$4
    shift 4
    local text start rc elapsed load
    text=$(quote "$@")
    if (( dry_run )); then
        echo "launch round=$round $name pause=$wait_s: $text"
        return 0
    fi
    if (( wait_s > 0 )); then
        sleep "$wait_s"
    fi
    read -r load _ < /proc/loadavg
    printf 'start\t%s\tround=%s\t%s\t%s\tload1=%s\tlogins=%s\tmhz=%s\ttemp_c=%s\tgpu=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round" "$name" "$text" "$load" \
        "$(logins)" "$(mean_mhz)" "$(cpu_temp)" \
        "$(gpu_query utilization.gpu,temperature.gpu,memory.used)" >> "$log"
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) round=$round $name" >> "$console"
    start=$EPOCHREALTIME
    # Output and errors go to the terminal and are appended to console.log.
    if [[ -n $stdout_file ]]; then
        # Standard output goes to the file (and then to console.log).
        if "$@" 2>&1 > "$stdout_file" | tee -a "$console" >&2; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
        tee -a "$console" < "$stdout_file"
    else
        if "$@" 2>&1 | tee -a "$console"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    fi
    elapsed=$(awk -v a="$start" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.1f", b - a }')
    printf 'end\t%s\tround=%s\t%s\texit=%s\telapsed_s=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round" "$name" "$rc" "$elapsed" >> "$log"
    return "$rc"
}

stop_after() {
    echo "sweep: $1 failed; stopping. Rows already written stay in $out." >&2
    exit 1
}

# 5. Timed rounds.
ef_list_for_round() {
    # The ef values rotated by two positions per round.
    local round=$1 start j list=""
    start=$(( 2 * (round - 1) % ${#ef_values[@]} ))
    for (( j = 0; j < ${#ef_values[@]}; j++ )); do
        list+="${ef_values[(start + j) % ${#ef_values[@]}]},"
    done
    echo "${list%,}"
}

# Configurations that take --query-count; with it, V1 on the full query set
# joins the rotation as two more configurations.
config_query=(1 1 1 1 1 1 1 1)
query_args=()
if [[ -n $query_count ]]; then
    query_args=(--query-count "$query_count")
    config_names+=("${v1full_names[@]}")
    config_args+=("${v1full_args[@]}")
    config_query+=(0 0)
fi

timed_launch() {
    # timed_launch ROUND NAME EF_LIST ARGS_TEXT [EXTRA_ARGS...]
    local round=$1 name=$2 efs=$3 args_text=$4
    shift 4
    local args
    read -r -a args <<< "$args_text"
    launch "$round" "$name" "$pause_s" "" \
        "$hvs" --data "$data" "${args[@]}" --ef "$efs" \
        --repetitions "$repetitions" --launch "$round" \
        --csv "$out/search.csv" --tag "$label" "$@" ||
        stop_after "$name (round $round)"
}

run_round() {
    # run_round ROUND [ONLY]: every configuration once, in the order this
    # round rotates to, or only those in the comma separated list ONLY.
    local round=$1 only_list=${2-} efs first i c extra
    efs=$(ef_list_for_round "$round")
    first=$(( 3 * (round - 1) % ${#config_names[@]} ))
    for (( i = 0; i < ${#config_names[@]}; i++ )); do
        c=$(( (first + i) % ${#config_names[@]} ))
        if [[ -n $only_list && ",$only_list," != *",${config_names[c]},"* ]]; then
            continue
        fi
        extra=()
        if (( config_query[c] )); then
            extra=("${query_args[@]}")
        fi
        timed_launch "$round" "${config_names[c]}" "$efs" "${config_args[c]}" \
            "${extra[@]}"
    done
}

# The pause before the first launch comes after the build and the tests.
if [[ -n $extend ]]; then
    # The extra round: the reason first, then the named configurations.
    extend_line=$(printf 'extend\t%s\tlaunch=%s\tonly=%s\treason=%s' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$extra_launch" "$only" "$reason")
    if (( dry_run )); then
        echo "+ append to $log: $extend_line"
    else
        printf '%s\n' "$extend_line" >> "$log"
    fi
    run_round "$extra_launch" "$only"
else
    for (( round = 1; round <= rounds; round++ )); do
        run_round "$round"
    done

    # 6. Verification launches on the full query set.
    all_efs=$(IFS=,; echo "${ef_values[*]}")
    for threads in 1 8; do
        launch verify "verify_t$threads" "$pause_s" "" \
            "$hvs" --data "$data" --mode hybrid-sync --backend cuda --verify \
            --repetitions 1 --launch 0 --ef "$all_efs" --threads "$threads" \
            --csv "$out/verify.csv" --tag "$label-verify" ||
            stop_after "verify_t$threads"
    done
fi

# 7. The results must match the expected hashes.
check=(python3 "$here/check_results.py" "$out" --expected "$here/expected_hashes.csv"
       --rounds "$rounds" --repetitions "$repetitions" --build-id "$id")
if (( full_queries > 0 )); then
    check+=(--full-queries "$full_queries")
fi
if [[ -n $query_count ]]; then
    check+=(--query-count "$query_count")
fi
if [[ -n $extend ]]; then
    check+=(--extra "$extra_launch:$only")
fi
if (( dry_run )); then
    step "${check[@]}"
else
    # Its output goes to the terminal and console.log; its first line, the
    # verdict, and its exit status also go to launches.log.
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) check_results.py" >> "$console"
    check_out=$("${check[@]}" 2>&1) && check_rc=0 || check_rc=$?
    if (( check_rc == 0 )); then
        printf '%s\n' "$check_out" | tee -a "$console"
    else
        printf '%s\n' "$check_out" | tee -a "$console" >&2
    fi
    printf 'check\t%s\texit=%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$check_rc" "${check_out%%$'\n'*}" >> "$log"
    (( check_rc == 0 )) || stop_after "check_results.py"
fi

# 8. Transfer micro-benchmark, three whole runs.
if (( ! skip_xfer )); then
    if (( ! dry_run )) && [[ ! -x $xfer ]]; then
        die "$xfer is not built (it needs a CUDA build); use --skip-xfer"
    fi
    for run in 1 2 3; do
        launch xfer "xfer_run$run" "$pause_s" "" \
            "$xfer" --data "$data" --csv "$out/xfer.csv" --tag "$label" --run "$run" ||
            stop_after "xfer_bench run $run"
    done
fi

# 9. Visit counts over the learn set; the binary files live in a temporary directory.
if (( ! skip_visits )); then
    if (( dry_run )); then
        tmp=/tmp/hvs-sweep.XXXXXX
    else
        tmp=$(mktemp -d "${TMPDIR:-/tmp}/hvs-sweep.XXXXXX")
    fi
    if (( learn_rows > 0 )); then
        half=$(( learn_rows / 2 ))
    else
        learn_rows=ROWS
        half=HALF
    fi
    counts=$tmp/counts
    texts=$out/visit_runs
    step mkdir "$counts" "$texts"

    visit_run() {
        # visit_run NAME EF [EXTRA_ARGS...]
        local name=$1 ef=$2
        shift 2
        launch visits "$name" 0 "$texts/$name.txt" \
            "$hvs" --data "$data" --ef "$ef" --threads 8 \
            --visit-counts "$counts/$name.bin" "$@" || stop_after "visit run $name"
    }
    visit_run all_ef16 16
    visit_run all_ef64 64
    visit_run all_ef256 256
    visit_run first_ef64 64 --visit-range "0,$half"
    visit_run second_ef64 64 --visit-range "$half,$half"

    step python3 "$here/visit_cdf.py" --counts-dir "$counts" --text-dir "$texts" \
        --learn-rows "$learn_rows" --out "$out/visit_cdf.csv"
fi

# 10. Faiss reference, then a last check that the sources did not change.
if [[ -z $extend && -e $data/faiss_reference.json ]]; then
    step cp "$data/faiss_reference.json" "$out/faiss_reference.json"
fi
step cmake --build "$build" || die "the final build failed"
if (( dry_run )); then
    echo "+ $hvs --build-id   (must print $id)"
    echo "dry run: nothing was run or written"
    exit 0
fi
final_id=$("$hvs" --build-id)
if [[ $final_id == "$id" ]]; then
    final_result=unchanged
else
    final_result=changed
fi
printf 'final_build_id\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$final_id" "$final_result" >> "$log"
if [[ $final_result == changed ]]; then
    die "the sources changed during the sweep: build id was $id, now $final_id; discard $out"
fi

echo "sweep complete, build id $id. Files written:"
find "$out" -type f | sort
