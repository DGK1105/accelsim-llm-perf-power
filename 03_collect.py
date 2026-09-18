#!/usr/bin/env python3
"""Turn a results dir from 02_run_sim.sh into tidy CSVs.

Usage: python3 03_collect.py /results/<run-name>

Reads (whatever exists):
  app_stats.csv, per_kernel_stats.csv, per_kernel_instance_stats.csv   get_stats.py output
  power__<app>__<args>.log                                             AccelWattch per-kernel reports
  stdout__<app>__<args>.txt                                            raw simulator stdout (fallback)

Writes (same dir):
  perf_tidy.csv   one row per (app, args, config, kernel, stat)
  power_tidy.csv  one row per (app, args, kernel instance, field)
  summary.csv     one row per (app, args, config): cycles, instructions, IPC, sim time, kernel
                  count, and average / max watts when a power report exists

Formats, taken from Accel-Sim v2.0.0 sources:
  get_stats.py (common.print_stat) emits blocks of
      ----------...,<commas>
      <stat regex>,<commas>
      APPS,<config>[,<config>...]
      <app>/<argfolder>--<kernel>,<value>[,<value>...]
  where <kernel> is "final_kernel" (whole run), "all_kernels" (build strings), a kernel name (-k),
  or "<kernel>--<n>" (-k -K). Values may be "NA".
  AccelWattch (gpgpu_sim_wrapper::print_power_kernel_stats) appends per kernel:
      kernel_name = <name>
      kernel_launch_uid = <uid>
      Kernel Average Power Data: / kernel_avg_power = W / gpu_avg_<component> = ... / gpu_tot_<counter> = ...
      Kernel Maximum Power Data: / kernel_max_power = W / ...
      Kernel Minimum Power Data: / kernel_min_power = W / ...
      Accumulative Power Statistics Over Previous Kernels: / gpu_tot_avg_power = W / gpu_tot_max_power / gpu_tot_min_power
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# AccelWattch writes component labels with a trailing comma ("gpu_avg_IBP, = 0.26"), hence the ",?".
KV = re.compile(r"^\s*([A-Za-z0-9_.\[\]-]+),?\s*=\s*(.*?)\s*$")


def to_num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def stat_label(regex: str) -> str:
    """gpu_tot_sim_cycle\\s*=\\s*(.*)  ->  gpu_tot_sim_cycle ; keeps the unit for the two rate stats."""
    name = re.split(r"\\s[*+]=|\s*=", regex, maxsplit=1)[0]
    name = re.sub(r"^(\\s[*+])+", "", name)          # leading \s+
    name = name.replace("\\", "").strip()
    if r"inst\/sec" in regex:
        name += "_inst_per_sec"
    elif r"cycle\/sec" in regex:
        name += "_cycle_per_sec"
    return name or regex


def parse_get_stats(path: Path, source: str):
    rows, stat, configs = [], None, None
    for raw in path.read_text().splitlines():
        line = raw.rstrip("\n")
        if not line.strip(","):
            continue
        if line.startswith("-----"):
            stat, configs = None, None            # block separator; next line is the stat regex
            continue
        cells = [c.strip() for c in line.split(",")]
        if stat is None:
            stat = line.rstrip(",").strip()
            continue
        if cells[0] == "APPS":
            configs = cells[1:]
            continue
        if configs is None or "--" not in cells[0]:
            continue
        appargs, kernel = cells[0].split("--", 1)   # first "--": instance rows are "<app>/<args>--<kernel>--<n>"
        app, _, args = appargs.partition("/")
        for cfg, val in zip(configs, cells[1:]):
            num = to_num(val)
            rows.append({
                "source": source, "app": app, "args": args, "config": cfg, "kernel": kernel,
                "stat": stat_label(stat), "value": "" if num is None else num, "raw": val,
                "stat_regex": stat,
            })
    return rows


def parse_power_log(path: Path):
    """One dict per kernel block; block starts at 'kernel_name = '."""
    blocks, cur = [], None
    for line in path.read_text().splitlines():
        m = KV.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key == "kernel_name":
            cur = {"kernel": val.strip() or f"kernel_{len(blocks)}", "kernel_idx": len(blocks), "fields": {}}
            blocks.append(cur)
            continue
        if cur is None:                          # values before any kernel header: keep, but flag
            cur = {"kernel": "kernel_0", "kernel_idx": 0, "fields": {}}
            blocks.append(cur)
        num = to_num(val)
        if num is not None:
            cur["fields"][key] = num
    return blocks


def parse_stdout(path: Path):
    """Last occurrence of the whole-run totals in raw simulator output (fallback when get_stats is absent)."""
    want = {"gpu_tot_sim_cycle": "cycles", "gpu_tot_sim_insn": "instructions", "gpu_tot_ipc": "ipc"}
    found = {}
    with path.open(errors="replace") as f:
        for line in f:
            m = KV.match(line)
            if m and m.group(1) in want:
                num = to_num(m.group(2))
                if num is not None:
                    found[want[m.group(1)]] = num
            elif line.startswith("gpgpu_simulation_time"):
                s = re.search(r"\((\d+) sec\)", line)
                if s:
                    found["sim_time_s"] = float(s.group(1))
    return found


def split_name(stem: str, prefix: str):
    body = stem[len(prefix):]
    app, _, args = body.partition("__")
    return app, args


def write_csv(path: Path, rows, cols):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(d: Path):
    if not d.is_dir():
        sys.exit(f"not a directory: {d}")

    # ---- performance -------------------------------------------------------------------------
    perf = []
    for name, source in (("app_stats.csv", "app"), ("per_kernel_stats.csv", "kernel"),
                         ("per_kernel_instance_stats.csv", "instance")):
        p = d / name
        if p.exists():
            perf += parse_get_stats(p, source)
    if perf:
        write_csv(d / "perf_tidy.csv", perf,
                  ["source", "app", "args", "config", "kernel", "stat", "value", "raw", "stat_regex"])

    # ---- power ---------------------------------------------------------------------------------
    power_rows, power_summary = [], {}
    for p in sorted(d.glob("power__*.log")):
        app, args = split_name(p.stem, "power__")
        blocks = parse_power_log(p)
        for b in blocks:
            for field, val in b["fields"].items():
                power_rows.append({"app": app, "args": args, "kernel_idx": b["kernel_idx"],
                                   "kernel": b["kernel"], "field": field, "value": val})
        if blocks:
            last = blocks[-1]["fields"]
            per_kernel_avg = [b["fields"]["kernel_avg_power"] for b in blocks if "kernel_avg_power" in b["fields"]]
            power_summary[(app, args)] = {
                # gpu_tot_avg_power in the last block is the sample-weighted average over the whole run
                "avg_power_W": last.get("gpu_tot_avg_power",
                                        sum(per_kernel_avg) / len(per_kernel_avg) if per_kernel_avg else ""),
                "max_power_W": last.get("gpu_tot_max_power",
                                        max((b["fields"].get("kernel_max_power", 0) for b in blocks), default="")),
                "power_kernels": len(blocks),
            }
    if power_rows:
        write_csv(d / "power_tidy.csv", power_rows, ["app", "args", "kernel_idx", "kernel", "field", "value"])

    # ---- summary -------------------------------------------------------------------------------
    want = {"gpu_tot_sim_cycle": "cycles", "gpu_tot_sim_insn": "instructions", "gpu_tot_ipc": "ipc",
            "gpgpu_simulation_time": "sim_time_s"}
    summ = defaultdict(dict)
    kernels = defaultdict(set)
    for r in perf:
        key = (r["app"], r["args"], r["config"])
        if r["source"] == "app" and r["kernel"] == "final_kernel" and r["stat"] in want and r["value"] != "":
            summ[key][want[r["stat"]]] = r["value"]
        if r["source"] == "kernel" and r["kernel"] not in ("final_kernel", "all_kernels"):
            kernels[key].add(r["kernel"])
        if r["source"] == "app" and r["stat"] in ("Accel-Sim-build", "GPGPU-Sim-build"):
            summ[key][r["stat"]] = r["raw"]
    for key, ks in kernels.items():
        summ[key]["kernels"] = len(ks)
    # Fallback: raw stdout when get_stats output is missing for an app
    for p in sorted(d.glob("stdout__*.txt")):
        app, args = split_name(p.stem, "stdout__")
        if not any(k[0] == app and k[1] == args for k in summ):
            vals = parse_stdout(p)
            if vals:
                summ[(app, args, "stdout")].update(vals)
    for key, v in summ.items():
        if v.get("cycles") and v.get("instructions") is not None:
            v["ipc_calc"] = v["instructions"] / v["cycles"]
        v.update(power_summary.get((key[0], key[1]), {}))

    cols = ["app", "args", "config", "cycles", "instructions", "ipc", "ipc_calc", "sim_time_s", "kernels",
            "avg_power_W", "max_power_W", "power_kernels", "Accel-Sim-build", "GPGPU-Sim-build"]
    if summ:
        out = [{"app": k[0], "args": k[1], "config": k[2], **v} for k, v in sorted(summ.items())]
        write_csv(d / "summary.csv", out, cols)
        # console view: the columns that fit
        show = ["app", "config", "cycles", "instructions", "ipc", "kernels", "avg_power_W"]
        widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in out)) for c in show}
        print("  ".join(c.ljust(widths[c]) for c in show))
        for r in out:
            print("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in show))
    print(f"\nperf rows: {len(perf)}  power rows: {len(power_rows)}  apps: {len(summ)}  -> {d}")
    if not perf and not power_rows:
        print("Nothing parsed. Expected app_stats.csv / per_kernel_stats.csv / power__*.log from 02_run_sim.sh.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
