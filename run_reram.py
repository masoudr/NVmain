#!/usr/bin/env python3
"""
NVMain ReRAM Graph Engine Simulation Runner
==========================================

  SIZE/SIZES  -- dimensions of each crossbar array (e.g. 8x8, 128x128).
                 Controls how many wordlines x bitlines a single crossbar has.
                 Sweeping this shows how energy/latency scales with array size.

  XBARS -- number of independent crossbar arrays in the engine.
           Controls engine-level parallelism (always a single integer).

Single run:
    python3 run_reram.py [config] [trace] [cycles] [PARAM=value ...]

Sweep over crossbar sizes:
    python3 run_reram.py [config] [trace] [cycles] SIZES=8x8,16x16,... [XBARS=N]

Defaults:
    config = Config/ReRAM_GraphEngine.config
    trace  = Tests/Traces/hello_world.nvt
    cycles = 0  (full trace)

Override syntax:
    SIZE=NxN     Single crossbar dimension (N must be a multiple of 8).
    SIZES=a,b,c  Comma-separated list of crossbar dimensions to sweep.
                   Example: SIZES=8x8,16x16,32x32,64x64,128x128
    XBARS=N      Number of crossbars per engine (single positive integer).
    CLK=N        Engine / memory clock in MHz.

Sweep output:
    Results are printed as a Python dict and written to sweep_results.py.
    All numeric values are stored at full float precision (no rounding).
    Dict key: results[size_str][xbars]

Examples:
    # Single run, defaults (128x128, 4 crossbars)
    python3 run_reram.py

    # Single run, specific size
    python3 run_reram.py Config/ReRAM_GraphEngine.config \\
        Tests/Traces/hello_world.nvt 0 SIZE=64x64 XBARS=4

    # Sweep crossbar sizes, fixed 4 crossbars per engine
    python3 run_reram.py Config/ReRAM_GraphEngine.config \\
        Tests/Traces/hello_world.nvt 0 \\
        SIZES=8x8,16x16,32x32,64x64,128x128,256x256,512x512,1024x1024 XBARS=4
"""

import sys
import subprocess
import re
import os
import pprint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE_WIDTH = 8   # DeviceWidth in the config; bits per device

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_config(config_path):
    """Return dict of numeric params from an NVMain config file."""
    params = {}
    with open(config_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith(';') or not s:
                continue
            m = re.match(r'^(\w+)\s*=?\s*([\d.eE+\-]+)', s)
            if m:
                try:
                    params[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return params


def parse_size_spec(spec):
    """
    'NxN'          -> [(N, N)]
    'AxA,BxB,CxC' -> [(A,A), (B,B), (C,C)]
    """
    parts = spec.strip().split(',')
    sizes = []
    for part in parts:
        m = re.fullmatch(r'(\d+)[xX](\d+)', part.strip())
        if not m:
            raise ValueError(f"Invalid size (expected NxN): {part!r}")
        r, c = int(m.group(1)), int(m.group(2))
        if r != c:
            raise ValueError(f"Only square crossbars are supported, got {r}x{c}.")
        if r % DEVICE_WIDTH != 0:
            raise ValueError(f"Crossbar size {r} must be a multiple of {DEVICE_WIDTH}.")
        sizes.append((r, c))
    return sizes

# ---------------------------------------------------------------------------
# Build NVMain override list for one (size, xbars) combination
# ---------------------------------------------------------------------------

def build_overrides(size, banks, base_params, passthrough):
    """
    size       : (N, N) tuple, or None to use config defaults
    xbars      : int or None to use config defaults (passed to NVMain as BANKS)
    base_params: dict from parse_config()
    passthrough: list of extra KEY=val strings to append verbatim
    Returns list of strings suitable for the NVMain command line.
    """
    n_ref       = int(base_params.get('COLS', 16)) * int(base_params.get('DeviceWidth', DEVICE_WIDTH))
    erd_ref     = base_params.get('Erd',     0.081200)
    ewr_ref     = base_params.get('Ewr',     1.684811)
    eopenrd_ref = base_params.get('Eopenrd', 0.001616)

    ovr = []
    if banks is not None:
        ovr.append(f"BANKS={banks}")
    if size is not None:
        N = size[0]
        ovr.append(f"ROWS={N}")
        ovr.append(f"MATHeight={N}")
        ovr.append(f"COLS={N // DEVICE_WIDTH}")
        scale = N / n_ref
        # Full double precision so NVMain's atof() gets an exact value
        ovr.append(f"Erd={erd_ref * scale:.17g}")
        ovr.append(f"Ewr={ewr_ref * scale:.17g}")
        ovr.append(f"Eopenrd={eopenrd_ref * scale:.17g}")
    ovr.extend(passthrough)
    return ovr

# ---------------------------------------------------------------------------
# Run one simulation
# ---------------------------------------------------------------------------

def run_nvmain(nvmain_bin, config, trace, cycles, overrides):
    """Run nvmain; return (stdout+stderr string, error_string_or_None)."""
    cmd = [nvmain_bin, config, trace, cycles] + overrides
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    if result.returncode != 0 and "NVMainTraceReader: Reached EOF" not in out:
        return None, out
    return out, None

# ---------------------------------------------------------------------------
# Parse NVMain output -> result dict (all values are exact floats / ints)
# ---------------------------------------------------------------------------

def parse_nvmain_output(output, ns_per_cycle):
    """
    Parse NVMain stdout/stderr and return a result dict.
    All latency/energy values are stored as full-precision Python floats.
    """
    avg_read_lat  = None
    avg_write_lat = None
    total_reads   = None
    total_writes  = None
    xbar_re       = {}   # bank_id -> nJ
    xbar_we       = {}
    xbar_reads    = {}
    xbar_writes   = {}

    for line in output.splitlines():
        line = line.strip()

        m = re.search(r'FRFCFS-WQF\.averageReadLatency\s+([\d.eE+\-]+)', line)
        if m:
            avg_read_lat = float(m.group(1))

        m = re.search(r'FRFCFS-WQF\.averageWriteLatency\s+([\d.eE+\-]+)', line)
        if m:
            avg_write_lat = float(m.group(1))

        m = re.search(r'totalReadRequests\s+(\d+)', line)
        if m:
            total_reads = int(m.group(1))

        m = re.search(r'totalWriteRequests\s+(\d+)', line)
        if m:
            total_writes = int(m.group(1))

        m = re.search(r'bank(\d+)\.subarray0\.readEnergyPerAccess\s+([\d.eE+\-]+)nJ', line)
        if m:
            xbar_re[int(m.group(1))] = float(m.group(2))

        m = re.search(r'bank(\d+)\.subarray0\.writeEnergyPerAccess\s+([\d.eE+\-]+)nJ', line)
        if m:
            xbar_we[int(m.group(1))] = float(m.group(2))

        m = re.search(r'bank(\d+)\.subarray0\.reads\s+(\d+)', line)
        if m:
            xbar_reads[int(m.group(1))] = int(m.group(2))

        m = re.search(r'bank(\d+)\.subarray0\.writes\s+(\d+)', line)
        if m:
            xbar_writes[int(m.group(1))] = int(m.group(2))

    num_xbars = len(xbar_re)
    avg_re = sum(xbar_re.values()) / num_xbars if num_xbars else None
    avg_we = sum(xbar_we.values()) / num_xbars if num_xbars else None

    per_crossbar = {
        b: {
            "reads":           xbar_reads.get(b, 0),
            "writes":          xbar_writes.get(b, 0),
            "read_energy_nj":  xbar_re.get(b),
            "write_energy_nj": xbar_we.get(b),
        }
        for b in sorted(set(xbar_re) | set(xbar_we))
    }

    return {
        "total_reads":           total_reads,
        "total_writes":          total_writes,
        "num_crossbars":         num_xbars,
        "read_latency_cycles":   avg_read_lat,
        "write_latency_cycles":  avg_write_lat,
        "read_latency_ns":       avg_read_lat  * ns_per_cycle if avg_read_lat  is not None else None,
        "write_latency_ns":      avg_write_lat * ns_per_cycle if avg_write_lat is not None else None,
        "read_energy_nj":        avg_re,
        "write_energy_nj":       avg_we,
        "per_crossbar":          per_crossbar,
    }

# ---------------------------------------------------------------------------
# Pretty-print a single result
# ---------------------------------------------------------------------------

def print_single_result(res, size_label, ns_per_cycle):
    sep  = "=" * 62
    sep2 = "-" * 62
    print(sep)
    print("  SIMULATION RESULTS")
    print(sep)
    tr = (res["total_reads"] or 0)
    tw = (res["total_writes"] or 0)
    print(f"\n  {'Total requests simulated':38s}  {tr+tw:>10,}")
    print(f"  {'  Reads':38s}  {tr:>10,}")
    print(f"  {'  Writes':38s}  {tw:>10,}")
    print(f"  {'Number of crossbars (XBARS)':38s}  {res['num_crossbars']:>10}")

    print(f"\n{sep2}")
    print("  ACCESS LATENCY  (controller average, all crossbars)")
    print(sep2)
    if res["read_latency_cycles"] is not None:
        print(f"  {'Read  latency':38s}  {res['read_latency_cycles']:>12.6f} cycles  =  {res['read_latency_ns']:>12.6f} ns")
    if res["write_latency_cycles"] is not None:
        print(f"  {'Write latency':38s}  {res['write_latency_cycles']:>12.6f} cycles  =  {res['write_latency_ns']:>12.6f} ns")

    print(f"\n{sep2}")
    print(f"  ENERGY PER ACCESS  (per {size_label} crossbar)")
    print(sep2)
    pc = res["per_crossbar"]
    if pc:
        print(f"\n  {'Crossbar':>10}  {'Reads':>8}  {'Writes':>8}  {'Read E/access':>20}  {'Write E/access':>20}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}  {'-'*20}")
        for b in sorted(pc):
            d = pc[b]
            print(f"  {b:>10}  {d['reads']:>8,}  {d['writes']:>8,}"
                  f"  {d['read_energy_nj']:>17.10f} nJ  {d['write_energy_nj']:>17.10f} nJ")
        avg_re = res["read_energy_nj"]
        avg_we = res["write_energy_nj"]
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}  {'-'*20}")
        print(f"  {'Average':>10}  {'':>8}  {'':>8}  {avg_re:>17.10f} nJ  {avg_we:>17.10f} nJ")

    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    if res["read_latency_ns"] and res["write_latency_ns"] and res["read_energy_nj"]:
        print(f"  Read  latency  :  {res['read_latency_cycles']:.6f} cycles  =  {res['read_latency_ns']:.6f} ns")
        print(f"  Write latency  :  {res['write_latency_cycles']:.6f} cycles  =  {res['write_latency_ns']:.6f} ns")
        print(f"  Read  energy   :  {res['read_energy_nj']:.10f} nJ  (avg across {res['num_crossbars']} crossbars)")
        print(f"  Write energy   :  {res['write_energy_nj']:.10f} nJ  (avg across {res['num_crossbars']} crossbars)")
    print(sep)
    print()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

positional    = [a for a in sys.argv[1:] if '=' not in a]
raw_overrides = [a for a in sys.argv[1:] if '=' in a]

config = positional[0] if len(positional) > 0 else "Config/ReRAM_GraphEngine.config"
trace  = positional[1] if len(positional) > 1 else "Tests/Traces/hello_world.nvt"
cycles = positional[2] if len(positional) > 2 else "0"

# Parse overrides: extract sweep dimensions and passthrough args
sweep_sizes  = None   # list of (N,N) tuples, or None
sweep_banks  = None   # list of ints, or None
passthrough  = []     # KEY=val args forwarded verbatim to NVMain

for arg in raw_overrides:
    key, _, val = arg.partition('=')
    ku = key.upper()
    if ku in ('SIZE', 'SIZES'):
        try:
            sweep_sizes = parse_size_spec(val)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    elif ku == 'XBARS':
        if not val.isdigit() or int(val) < 1:
            print(f"ERROR: XBARS must be a positive integer, got: {val!r}")
            sys.exit(1)
        sweep_banks = [int(val)]
    else:
        passthrough.append(arg)

# Sweep mode = more than one crossbar size to simulate
is_sweep = sweep_sizes is not None and len(sweep_sizes) > 1

# Default to single dimension if only one value given
if sweep_sizes is None:
    sweep_sizes = [None]    # None → use config defaults
if sweep_banks is None:
    sweep_banks = [None]    # None → use config defaults

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

nvmain = "./nvmain.fast"
for path, label in [(nvmain, "nvmain.fast"), (config, "config"), (trace, "trace")]:
    if not os.path.exists(path):
        print(f"ERROR: {label} not found: {path}")
        if label == "nvmain.fast":
            print("       Build first with: scons --build-type=fast")
        sys.exit(1)

base_params  = parse_config(config)
clk_mhz      = int(base_params.get('CLK', 400))
for arg in raw_overrides:
    m = re.fullmatch(r'CLK=(\d+)', arg)
    if m:
        clk_mhz = int(m.group(1))
ns_per_cycle = 1000.0 / clk_mhz

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------

# results dict: { "NxN": { xbars: result_dict } }
results = {}

total_runs = len(sweep_sizes) * len(sweep_banks)
run_idx    = 0

for size in sweep_sizes:
    size_label = f"{size[0]}x{size[1]}" if size else "default"
    for banks in sweep_banks:
        run_idx += 1
        ovr = build_overrides(size, banks, base_params, passthrough)

        if is_sweep:
            xbar_str = f"{banks} crossbar{'s' if banks != 1 else ''}" if banks else "default crossbars"
            print(f"[{run_idx}/{total_runs}] SIZE={size_label}  XBARS={xbar_str} ...", flush=True)
        else:
            print("=" * 62)
            print("  NVMain ReRAM Simulation")
            print("=" * 62)
            print(f"  Config : {config}")
            print(f"  Trace  : {trace}")
            print(f"  Cycles : {'full trace' if cycles == '0' else cycles}")
            print(f"  Clock  : {clk_mhz} MHz  ({ns_per_cycle:.6f} ns/cycle)")
            if size:
                N = size[0]
                n_ref = int(base_params.get('COLS', 16)) * int(base_params.get('DeviceWidth', DEVICE_WIDTH))
                scale = N / n_ref
                erd_ref     = base_params.get('Erd',     0.081200)
                ewr_ref     = base_params.get('Ewr',     1.684811)
                eopenrd_ref = base_params.get('Eopenrd', 0.001616)
                print(f"  SIZE   : {size_label}  (ROWS={N}, MATHeight={N}, COLS={N//DEVICE_WIDTH})")
                print(f"           Energy scale {scale:.17g}x vs {n_ref}-bit reference crossbar")
                print(f"           Erd={erd_ref*scale:.17g} nJ  Ewr={ewr_ref*scale:.17g} nJ"
                      f"  Eopenrd={eopenrd_ref*scale:.17g} nJ")
            if banks:
                print(f"  XBARS  : {banks}  (passed to NVMain as BANKS={banks})")
            for p in passthrough:
                print(f"  Override: {p}")
            print("=" * 62)
            print("  Running simulation ...")
            print()

        raw_out, err = run_nvmain(nvmain, config, trace, cycles, ovr)
        if raw_out is None:
            print(f"ERROR: nvmain failed for SIZE={size_label} XBARS={banks}:")
            print(err[-2000:])
            if is_sweep:
                continue
            else:
                sys.exit(1)

        res = parse_nvmain_output(raw_out, ns_per_cycle)

        if not is_sweep:
            print_single_result(res, size_label, ns_per_cycle)

        banks_key = banks if banks is not None else int(base_params.get('BANKS', 4))
        if size_label not in results:
            results[size_label] = {}
        results[size_label][banks_key] = res

# ---------------------------------------------------------------------------
# Sweep summary table
# ---------------------------------------------------------------------------

if is_sweep:
    print()
    sep = "=" * 90
    print(sep)
    print("  SWEEP RESULTS SUMMARY")
    print(sep)
    hdr = f"  {'Size':>10}  {'XBars':>5}  {'RdLat(cyc)':>14}  {'RdLat(ns)':>12}  {'WrLat(cyc)':>14}  {'WrLat(ns)':>12}  {'RdE(nJ)':>16}  {'WrE(nJ)':>16}"
    print(hdr)
    print("  " + "-" * 86)
    for sz in results:
        for bk in sorted(results[sz]):
            r = results[sz][bk]
            rl_c = r["read_latency_cycles"]  or 0.0
            rl_n = r["read_latency_ns"]      or 0.0
            wl_c = r["write_latency_cycles"] or 0.0
            wl_n = r["write_latency_ns"]     or 0.0
            re   = r["read_energy_nj"]       or 0.0
            we   = r["write_energy_nj"]      or 0.0
            print(f"  {sz:>10}  {bk:>5}  {rl_c:>14.6f}  {rl_n:>12.6f}  {wl_c:>14.6f}  {wl_n:>12.6f}  {re:>16.10f}  {we:>16.10f}")
    print(sep)
    print()

# ---------------------------------------------------------------------------
# Python dict output  (always, single or sweep)
# All values are full-precision Python floats / ints; no formatting applied.
# ---------------------------------------------------------------------------

print()
print("# " + "=" * 62)
print("# Python results dict")
print("# " + "=" * 62)
print("results = \\")
print(pprint.pformat(results, sort_dicts=False))
print()

# Write to file
out_path = "nvmain_results.py"
with open(out_path, "w") as f:
    f.write("# NVMain ReRAM sweep results\n")
    f.write(f"# Config : {config}\n")
    f.write(f"# Trace  : {trace}\n")
    f.write(f"# CLK    : {clk_mhz} MHz  ({ns_per_cycle:.17g} ns/cycle)\n")
    f.write("# Keys   : results[size_str][num_banks]\n")
    f.write("#\n")
    f.write("results = \\\n")
    f.write(pprint.pformat(results, sort_dicts=False))
    f.write("\n")

print(f"Results written to: {out_path}")
