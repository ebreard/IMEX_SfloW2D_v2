#!/usr/bin/env python3
# Author: ECP_BREARD
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_inp(inp_path: Path):
    txt = inp_path.read_text(encoding="utf-8", errors="ignore")
    def ffloat(key, default):
        m = re.search(rf"{key}\s*=\s*([0-9\.\+EeDd-]+)", txt)
        return float(m.group(1).replace("D","E")) if m else default
    def fstr(key, default):
        m = re.search(rf'{key}\s*=\s*"(.*?)"', txt)
        return m.group(1) if m else default
    PRES = ffloat("PRES", 101300.0)
    T0   = ffloat("T_START", 0.0)
    DT   = ffloat("DT_OUTPUT", 1.0)
    RUN  = fstr("RUN_NAME", "run")
    return PRES, T0, DT, RUN

def nc_read_array(nc_path: Path, var_hint: str=None):
    """Return (varname, data, time) with data shaped (nt, ny, nx).
       Tries netCDF4 first, then h5py."""
    varname = None; time = None
    try:
        from netCDF4 import Dataset
        ds = Dataset(nc_path.as_posix(), "r")
        if var_hint:
            candidates = [var_hint]
        else:
            candidates = [n for n in ds.variables.keys()
                          if ("pore" in n.lower() and "pres" in n.lower()) or ("porepres" in n.lower())]
            if not candidates:
                candidates = [n for n in ds.variables.keys() if "pres" in n.lower()]
        chosen = None
        for name in candidates:
            v = ds.variables.get(name)
            if v is None: continue
            if hasattr(v, "dimensions") and any("time" in d.lower() for d in v.dimensions):
                chosen = name; break
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None:
            raise RuntimeError("No pore pressure variable found (netCDF4).")
        varname = chosen
        v = ds.variables[varname]
        data = v[:]
        if "time" in ds.variables:
            time = np.array(ds.variables["time"][:], dtype=float)
        ds.close()
        arr = np.array(data, dtype=float)
        if arr.ndim == 2: arr = arr[None, ...]
        return varname, arr, time
    except Exception:
        pass
    try:
        import h5py
        f = h5py.File(nc_path.as_posix(), "r")
        def all_dsets(h):
            for k, v in h.items():
                if isinstance(v, h5py.Dataset): yield v.name, v
                elif isinstance(v, h5py.Group): yield from all_dsets(v)
        dtarget = None
        if var_hint:
            for name, dset in all_dsets(f):
                if name.split("/")[-1] == var_hint:
                    dtarget = dset; varname = var_hint; break
        if dtarget is None:
            for name, dset in all_dsets(f):
                low = name.lower()
                if ("pore" in low and "pres" in low) or ("porepres" in low):
                    dtarget = dset; varname = name.split("/")[-1]; break
        if dtarget is None:
            for name, dset in all_dsets(f):
                if "pres" in name.lower():
                    dtarget = dset; varname = name.split("/")[-1]; break
        if dtarget is None:
            raise RuntimeError("No pore pressure dataset found (h5py).")
        data = np.array(dtarget[...], dtype=float)
        time = None
        if "time" in f.keys():
            try:
                time = np.array(f["time"][...], dtype=float).reshape(-1)
            except Exception:
                time = None
        f.close()
        if data.ndim == 2: data = data[None, ...]
        return varname, data, time
    except Exception as e:
        raise SystemExit(f"Failed to read NetCDF: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nc", type=str, help="Path to NetCDF file (e.g., example2D.nc)")
    ap.add_argument("--var", type=str, default=None, help="Variable name to read (default: auto-detect)")
    ap.add_argument("--outbase", type=str, default=None, help="Output base name (default: NC stem)")
    args = ap.parse_args()

    nc_path = Path(args.nc)
    if not nc_path.exists():
        raise SystemExit(f"NC file not found: {nc_path}")

    cwd = Path.cwd()
    inp = cwd / "IMEX_SfloW2D.inp"
    if not inp.exists():
        raise SystemExit("IMEX_SfloW2D.inp not found in current directory.")
    PRES, T0, DT, RUN = parse_inp(inp)

    varname, arr, time = nc_read_array(nc_path, args.var)
    nt = arr.shape[0]
    if time is None or len(time) != nt:
        time = T0 + np.arange(nt) * DT

    mean_absP = arr.reshape(nt, -1).mean(axis=1)
    mean_excess = mean_absP - PRES
    P0_mean = mean_absP[0]
    excess0 = mean_excess[0]

    base = args.outbase if args.outbase else nc_path.stem
    png = Path(f"{base}_pressure_summary.png")
    csv = Path(f"{base}_excess_timeseries.csv")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1); ax1.axis("off")
    ax1.text(0.0, 1.0,
             "\n".join([
                f"Run: {RUN}",
                f"NetCDF: {nc_path.name}",
                f"Variable: {varname}",
                "",
                f"t = 0",
                f"Domain-mean absolute P: {P0_mean:.3f} Pa",
                f"Ambient PRES (INP):     {PRES:.3f} Pa",
                f"Mean excess P - PRES:   {excess0:.3f} Pa",
             ]),
             va="top", ha="left", fontsize=12)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(time, mean_excess, linewidth=2)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("mean(P - PRES) (Pa)")
    ax2.set_title("Domain-mean excess pore pressure")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)

    with open(csv, "w", encoding="utf-8") as f:
        f.write("time_s,mean_abs_Pa,mean_excess_Pa\n")
        for ti, pa, pe in zip(time, mean_absP, mean_excess):
            f.write(f"{ti:.12g},{pa:.12g},{pe:.12g}\n")

    print(f"Saved PNG: {png}")
    print(f"Saved CSV: {csv}")
    print(f"[info] Using var='{varname}', PRES={PRES} Pa, nt={nt}, shape(t0)={arr[0].shape}")

if __name__ == "__main__":
    main()
