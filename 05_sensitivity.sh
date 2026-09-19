#!/usr/bin/env bash
# Run this INSIDE the container (cwd = /accel-sim), after 02_run_sim.sh has replayed the LLM trace once.
# One-parameter-at-a-time sensitivity sweep of the H100 model on the LLM layer: which hardware/model
# parameter does the decoder block actually care about?
#
# Usage: ./05_sensitivity.sh [trace-set] [passes]
#   trace-set : name under /traces (default qwen25-0.5b__layer12-fixed)
#   passes    : how many 12-kernel passes to replay (default 2 = prefill + first decode pass; the seven
#               decode passes agree within 1%, so one is representative and keeps a run at ~5 min)
#
# Variants are modifiers in util/job_launching/configs/define-sweep-cfgs.yml (the launcher globs
# configs/define-*.yml). extra_params are appended after the base gpgpusim.config, and GPGPU-Sim's option
# parser lets the last occurrence win; the script checks that in each run's stdout instead of assuming it.
set -euo pipefail
cd "${ACCELSIM_ROOT:-/accel-sim}"
SET="${1:-qwen25-0.5b__layer12-fixed}"; PASSES="${2:-2}"; KPP=12
SRC="/traces/$SET/llm_layer/layer/traces"; SUB="${SET}-p${PASSES}"; DST="/traces/$SUB/llm_layer/layer/traces"
[[ -f "$SRC/kernelslist.g" ]] || { echo "no $SRC/kernelslist.g"; exit 1; }

echo "== 1. Sweep modifiers"
cat > util/job_launching/configs/define-sweep-cfgs.yml <<'EOF'
# Written by 05_sensitivity.sh. One parameter each, relative to SM90_H100/gpgpusim.config
# (kernel_launch_latency 3000, dram_latency 194, l2_rop_latency 237, l1_latency 41, smem_latency 29).
LAUNCH0:
    extra_params: "-gpgpu_kernel_launch_latency 0"
LAUNCH1500:
    extra_params: "-gpgpu_kernel_launch_latency 1500"
DRAMLAT2X:
    extra_params: "-dram_latency 388"
DRAMLATHALF:
    extra_params: "-dram_latency 97"
L2LAT2X:
    extra_params: "-gpgpu_l2_rop_latency 474"
L1LAT2X:
    extra_params: "-gpgpu_l1_latency 82"
SMEMLAT2X:
    extra_params: "-gpgpu_smem_latency 58"
EOF
declare -A KEY=( [LAUNCH0]="-gpgpu_kernel_launch_latency" [LAUNCH1500]="-gpgpu_kernel_launch_latency" [DRAMLAT2X]="-dram_latency"
                 [DRAMLATHALF]="-dram_latency" [L2LAT2X]="-gpgpu_l2_rop_latency" [L1LAT2X]="-gpgpu_l1_latency" [SMEMLAT2X]="-gpgpu_smem_latency" )
CONFIGS=(H100-SASS H100-SASS-LAUNCH0 H100-SASS-LAUNCH1500 H100-SASS-DRAMLAT2X H100-SASS-DRAMLATHALF H100-SASS-L2LAT2X
         H100-SASS-L1LAT2X H100-SASS-SMEMLAT2X H200-SASS)

echo "== 2. Trace subset: first $PASSES pass(es) = $((PASSES * KPP)) kernels -> /traces/$SUB"
mkdir -p "$DST"
awk -v n=$((PASSES * KPP)) '/\.trace/ { if (++k > n) exit } { print }' "$SRC/kernelslist.g" > "$DST/kernelslist.g"
for f in $(grep -oE 'kernel-[^ ]+\.tracez' "$DST/kernelslist.g"); do ln -sf "$SRC/$f" "$DST/$f"; done
echo "   kernels listed: $(grep -c '\.tracez' "$DST/kernelslist.g")"

echo "== 3. Runs (sequential; each is a separate launch so a failure does not stop the sweep)"
OUT="/results/sweep-$SUB"; mkdir -p "$OUT"
for c in "${CONFIGS[@]}"; do
  name="sweep-$SUB-$c"
  if [[ -s "/results/$name/app_stats.csv" ]] && grep -q 'gpu_tot_sim_cycle' "/results/$name/app_stats.csv"; then echo "-- $c: already done"; continue; fi
  echo "-- $c"
  ./02_run_sim.sh llm_layer "/traces/$SUB" "$c" "$name" > "$OUT/run-$c.log" 2>&1 || echo "   FAILED (see $OUT/run-$c.log)"
done

echo "== 4. Summary -> $OUT/sensitivity.csv"
python3 - "$SUB" "$OUT" "$KPP" "$PASSES" <<'EOF'
import csv, glob, os, re, sys
sub, out, kpp, passes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = []
for d in sorted(glob.glob(f"/results/sweep-{sub}-*")):
    cfg = os.path.basename(d)[len(f"sweep-{sub}-"):]
    so = glob.glob(os.path.join(d, "stdout__*.txt"))
    if not so: rows.append(dict(config=cfg, status="no stdout")); continue
    cyc, params = [], {}
    for l in open(so[0], errors="replace"):
        if l.startswith("gpu_sim_cycle ="): cyc.append(int(l.split("=")[1]))
        m = re.match(r"^(-gpgpu_kernel_launch_latency|-dram_latency|-gpgpu_l2_rop_latency|-gpgpu_l1_latency|-gpgpu_smem_latency|-gpgpu_n_mem)\s+(\S+)", l)
        if m: params[m.group(1)] = m.group(2)          # the option dump prints the value actually in effect
    ok = len(cyc) == kpp * passes
    r = dict(config=cfg, status="ok" if ok else f"{len(cyc)} kernels", total=sum(cyc), prefill=sum(cyc[:kpp]),
             decode=sum(cyc[kpp:2 * kpp]) if passes > 1 else "", **{k.lstrip("-"): v for k, v in params.items()})
    for i, c in enumerate(cyc[:kpp * passes]): r[f"k{i + 1}"] = c
    rows.append(r)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(os.path.join(out, "sensitivity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
base = next((r for r in rows if r["config"] == "H100-SASS" and r.get("status") == "ok"), None)
print(f"{'config':26s} {'status':10s} {'prefill':>9s} {'decode':>9s} {'total':>9s}  vs H100-SASS")
for r in rows:
    if r.get("status") != "ok": print(f"{r['config']:26s} {r.get('status','?')}"); continue
    d = f"{(r['total'] / base['total'] - 1) * 100:+.1f}%" if base else ""
    print(f"{r['config']:26s} {r['status']:10s} {r['prefill']:9d} {r['decode']:9} {r['total']:9d}  {d}")
EOF
