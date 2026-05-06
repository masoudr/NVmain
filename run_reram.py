#!/usr/bin/env python3
"""
NVMain ReRAM Graph Engine Simulation Runner
Usage:  python3 run_reram.py [config] [trace] [cycles] [PARAM=value ...]

Defaults:
  config  = Config/RRAM_GraphEngine_128x128.config
  trace   = Tests/Traces/hello_world.nvt
  cycles  = 0  (simulate full trace)

Special PARAM overrides (handled by this script, not passed raw to NVMain):
  SIZE=NxN   Crossbar dimensions, e.g. SIZE=32x32, SIZE=128x128, SIZE=256x256
             N must be a multiple of 8 (DeviceWidth).
             Sets ROWS=N, MATHeight=N, COLS=N/8 automatically.

Standard NVMain overrides (passed directly):
  BANKS=N    Number of crossbars per engine
  CLK=N      Clock frequency in MHz

Examples:
  python3 run_reram.py
  python3 run_reram.py Config/RRAM_GraphEngine_128x128.config Tests/Traces/hello_world.nvt
  python3 run_reram.py Config/RRAM_GraphEngine_128x128.config Tests/Traces/hello_world.nvt 0 BANKS=8
  python3 run_reram.py Config/RRAM_GraphEngine_128x128.config Tests/Traces/hello_world.nvt 0 SIZE=32x32
  python3 run_reram.py Config/RRAM_GraphEngine_128x128.config Tests/Traces/hello_world.nvt 0 SIZE=256x256 BANKS=8
"""

import sys
import subprocess
import re
import os

# ---------------------------------------------------------------------------
# Arguments
# Positional: config, trace, cycles
# Any arg containing '=' is treated as a PARAM=value override
# SIZE=NxN is special: converted to ROWS/COLS/MATHeight overrides for NVMain
# ---------------------------------------------------------------------------
positional = [a for a in sys.argv[1:] if '=' not in a]
raw_overrides = [a for a in sys.argv[1:] if '=' in a]

config = positional[0] if len(positional) > 0 else "Config/RRAM_GraphEngine_128x128.config"
trace  = positional[1] if len(positional) > 1 else "Tests/Traces/hello_world.nvt"
cycles = positional[2] if len(positional) > 2 else "0"

DEVICE_WIDTH = 8  # bits per device; matches DeviceWidth in the config

# Parse SIZE=NxN if present
crossbar_size = None   # tuple (rows, cols) if specified
overrides = []         # final list passed to NVMain
for arg in raw_overrides:
    key, _, val = arg.partition('=')
    if key.upper() == 'SIZE':
        m = re.fullmatch(r'(\d+)[xX](\d+)', val)
        if not m:
            print(f"ERROR: SIZE must be NxN (e.g. SIZE=128x128), got: {val}")
            sys.exit(1)
        rows = int(m.group(1))
        cols_bits = int(m.group(2))
        if rows != cols_bits:
            print(f"ERROR: Only square crossbars are supported (got {rows}x{cols_bits}).")
            sys.exit(1)
        if rows % DEVICE_WIDTH != 0:
            print(f"ERROR: Crossbar size {rows} must be a multiple of DeviceWidth ({DEVICE_WIDTH}).")
            sys.exit(1)
        cols_addr = rows // DEVICE_WIDTH   # column addresses = bitlines / DeviceWidth
        crossbar_size = (rows, cols_bits)
        # Inject the three derived NVMain parameters
        overrides.append(f"ROWS={rows}")
        overrides.append(f"MATHeight={rows}")
        overrides.append(f"COLS={cols_addr}")
    else:
        overrides.append(arg)

nvmain = "./nvmain.fast"
if not os.path.exists(nvmain):
    print("ERROR: nvmain.fast not found. Build first with: scons --build-type=fast")
    sys.exit(1)
if not os.path.exists(config):
    print(f"ERROR: config file not found: {config}")
    sys.exit(1)
if not os.path.exists(trace):
    print(f"ERROR: trace file not found: {trace}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parse all relevant parameters from the config file
# ---------------------------------------------------------------------------
config_params = {}
with open(config) as f:
    for line in f:
        s = line.strip()
        if s.startswith(';') or not s:
            continue
        m = re.match(r'^(\w+)\s*=?\s*([\d.eE+\-]+)', s)
        if m:
            try:
                config_params[m.group(1)] = float(m.group(2))
            except ValueError:
                pass

erd_ref     = config_params.get('Erd',         0.081200)
ewr_ref     = config_params.get('Ewr',         1.684811)
eopenrd_ref = config_params.get('Eopenrd',     0.001616)
cols_ref    = int(config_params.get('COLS',        16))
devwidth    = int(config_params.get('DeviceWidth',  8))
n_ref       = cols_ref * devwidth   # reference bitlines per crossbar row (128 for 128x128)
clk_mhz     = int(config_params.get('CLK',        400))

# Apply CLK= command-line override if present
for arg in raw_overrides:
    m = re.fullmatch(r'CLK=(\d+)', arg)
    if m:
        clk_mhz = int(m.group(1))
ns_per_cycle = 1000.0 / clk_mhz

# If SIZE was specified, inject scaled energy overrides now that we know the reference values.
# For a crossbar, read/write energy scales linearly with the number of bitlines (= N for NxN).
if crossbar_size:
    N = crossbar_size[0]
    scale = N / n_ref
    overrides.append(f"Erd={erd_ref * scale:.6f}")
    overrides.append(f"Ewr={ewr_ref * scale:.6f}")
    overrides.append(f"Eopenrd={eopenrd_ref * scale:.6f}")

# Determine crossbar label for reporting
if crossbar_size:
    xbar_label = f"{crossbar_size[0]}x{crossbar_size[1]}"
else:
    m = re.search(r'(\d+)x(\d+)', os.path.basename(config))
    xbar_label = f"{m.group(1)}x{m.group(2)}" if m else "NxN"

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
cmd = [nvmain, config, trace, cycles] + overrides
print("=" * 60)
print("  NVMain ReRAM Simulation")
print("=" * 60)
print(f"  Config : {config}")
print(f"  Trace  : {trace}")
print(f"  Cycles : {'full trace' if cycles == '0' else cycles}")
print(f"  Clock  : {clk_mhz} MHz  ({ns_per_cycle:.2f} ns/cycle)")
if overrides:
    for ov in overrides:
        # Show the SIZE= form if we expanded it, otherwise show raw
        if ov.startswith('ROWS=') or ov.startswith('MATHeight=') or ov.startswith('COLS='):
            continue   # these come from SIZE=, already shown below
        print(f"  Override: {ov}")
    if crossbar_size:
        N = crossbar_size[0]
        scale = N / n_ref
        print(f"  Override: SIZE={xbar_label}  →  ROWS={N}, MATHeight={N}, COLS={N//DEVICE_WIDTH}")
        print(f"            Energy scaled {scale:.4g}x vs reference {n_ref}-bitline crossbar")
        print(f"            Erd={erd_ref*scale:.6f} nJ  Ewr={ewr_ref*scale:.6f} nJ  Eopenrd={eopenrd_ref*scale:.6f} nJ")
print("=" * 60)
print("  Running simulation ...")
print()

result = subprocess.run(cmd, capture_output=True, text=True)
output = result.stdout + result.stderr

if result.returncode != 0 and "NVMainTraceReader: Reached EOF" not in output:
    print("ERROR: nvmain exited with an error:")
    print(output[-2000:])
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parse output
# ---------------------------------------------------------------------------
avg_read_lat   = None
avg_write_lat  = None
total_reads    = None
total_writes   = None

# Per-crossbar (subarray0 of each bank)
crossbar_read_energy  = {}   # bank_id -> nJ
crossbar_write_energy = {}   # bank_id -> nJ
crossbar_reads        = {}
crossbar_writes       = {}

for line in output.splitlines():
    line = line.strip()

    # Controller-level latency
    m = re.search(r'FRFCFS-WQF\.averageReadLatency\s+([\d.]+)', line)
    if m:
        avg_read_lat = float(m.group(1))

    m = re.search(r'FRFCFS-WQF\.averageWriteLatency\s+([\d.]+)', line)
    if m:
        avg_write_lat = float(m.group(1))

    # Total requests
    m = re.search(r'totalReadRequests\s+(\d+)', line)
    if m:
        total_reads = int(m.group(1))

    m = re.search(r'totalWriteRequests\s+(\d+)', line)
    if m:
        total_writes = int(m.group(1))

    # Per-crossbar energy (subarray0 only = one crossbar per bank)
    m = re.search(r'bank(\d+)\.subarray0\.readEnergyPerAccess\s+([\d.e+]+)nJ', line)
    if m:
        crossbar_read_energy[int(m.group(1))] = float(m.group(2))

    m = re.search(r'bank(\d+)\.subarray0\.writeEnergyPerAccess\s+([\d.e+]+)nJ', line)
    if m:
        crossbar_write_energy[int(m.group(1))] = float(m.group(2))

    m = re.search(r'bank(\d+)\.subarray0\.reads\s+(\d+)', line)
    if m:
        crossbar_reads[int(m.group(1))] = int(m.group(2))

    m = re.search(r'bank(\d+)\.subarray0\.writes\s+(\d+)', line)
    if m:
        crossbar_writes[int(m.group(1))] = int(m.group(2))

num_crossbars = len(crossbar_read_energy)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
sep  = "=" * 60
sep2 = "-" * 60

print(sep)
print("  SIMULATION RESULTS")
print(sep)

print(f"\n  {'Total requests simulated':35s}  {(total_reads or 0) + (total_writes or 0):>10,}")
print(f"  {'  Reads':35s}  {total_reads or 0:>10,}")
print(f"  {'  Writes':35s}  {total_writes or 0:>10,}")
print(f"  {'Number of crossbars (BANKS)':35s}  {num_crossbars:>10}")

# --- Latency ---
print(f"\n{sep2}")
print("  ACCESS LATENCY  (controller average, all crossbars)")
print(sep2)
if avg_read_lat is not None:
    print(f"  {'Read  latency':35s}  {avg_read_lat:>8.2f} cycles  = {avg_read_lat * ns_per_cycle:>8.2f} ns")
if avg_write_lat is not None:
    print(f"  {'Write latency':35s}  {avg_write_lat:>8.2f} cycles  = {avg_write_lat * ns_per_cycle:>8.2f} ns")

# --- Energy per access ---
print(f"\n{sep2}")
print(f"  ENERGY PER ACCESS  (per {xbar_label} crossbar)")
print(sep2)

if crossbar_read_energy:
    print(f"\n  {'Crossbar':>10}  {'Reads':>8}  {'Writes':>8}  {'Read E/access':>16}  {'Write E/access':>16}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*16}  {'-'*16}")
    all_re = []
    all_we = []
    for b in sorted(crossbar_read_energy):
        re_nj = crossbar_read_energy[b]
        we_nj = crossbar_write_energy.get(b, 0.0)
        nr    = crossbar_reads.get(b, 0)
        nw    = crossbar_writes.get(b, 0)
        all_re.append(re_nj)
        all_we.append(we_nj)
        print(f"  {b:>10}  {nr:>8,}  {nw:>8,}  {re_nj:>13.4f} nJ  {we_nj:>13.4f} nJ")

    avg_re = sum(all_re) / len(all_re)
    avg_we = sum(all_we) / len(all_we)
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*16}  {'-'*16}")
    print(f"  {'Average':>10}  {'':>8}  {'':>8}  {avg_re:>13.4f} nJ  {avg_we:>13.4f} nJ")

print(f"\n{sep}")
print("  SUMMARY")
print(sep)
if avg_read_lat and avg_write_lat and crossbar_read_energy:
    avg_re = sum(crossbar_read_energy.values()) / len(crossbar_read_energy)
    avg_we = sum(crossbar_write_energy.values()) / len(crossbar_write_energy)
    print(f"  Read  latency  :  {avg_read_lat:.2f} cycles  =  {avg_read_lat * ns_per_cycle:.2f} ns")
    print(f"  Write latency  :  {avg_write_lat:.2f} cycles  =  {avg_write_lat * ns_per_cycle:.2f} ns")
    print(f"  Read  energy   :  {avg_re:.4f} nJ  per access  (avg across {num_crossbars} crossbars)")
    print(f"  Write energy   :  {avg_we:.4f} nJ  per access  (avg across {num_crossbars} crossbars)")
print(sep)
print()
