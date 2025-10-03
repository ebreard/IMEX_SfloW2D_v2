#!/usr/bin/env python3
# Author: ECP Breard
# Excess pore pressure comparator (v3, fixed quotes):
# - Pre-t2: p_excess_expected(t) = rho_mix * g * PPF * h(t) on 2x2 source patch
# - Post-t2: exponential decay from value at t2 with lambda = (pi/2)^2 * D / h(t2)^2
# - Also plots & saves h(t), rho_mix, g, and PPF over time

import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- user tunables ----
D_DEFAULT = 0.01     # m^2/s
VAR_HINT_PRESS = None  # e.g. "pore_pressure"
VAR_HINT_H = None      # e.g. "h"
RHO_MIX_OVERRIDE = None  # set via --rho_mix to force a value (kg/m^3)

def f2float(s: str) -> float:
    return float(str(s).replace("D","E").replace("d","E"))

def parse_inp(inp_path: Path):
    txt = inp_path.read_text(encoding="utf-8", errors="ignore")
    def ffloat(key, default=None):
        m = re.search(rf"{key}\s*=\s*([0-9\.\+EeDd-]+)", txt)
        return f2float(m.group(1)) if m else default
    PRES = ffloat("PRES", 101300.0)
    X0 = ffloat("X0", None); Y0 = ffloat("Y0", None)
    DX = ffloat("CELL_SIZE", None)
    NX = int(ffloat("COMP_CELLS_X", 0)); NY = int(ffloat("COMP_CELLS_Y", 0))
    XS = ffloat("X_SOURCE", None); YS = ffloat("Y_SOURCE", None)
    # TIME_PARAM: use 2nd value as t2
    t2 = None
    m = re.search(r"TIME_PARAM\s*=\s*([^/\n]+)", txt)
    if m:
        vals = [f2float(v) for v in re.split(r"[,\s]+", m.group(1).strip()) if v.strip()]
        if len(vals) >= 2: t2 = vals[1]
    # physics
    PPF = ffloat("PORE_PRES_FRACT", 0.0)
    ALPHAS = ffloat("ALPHAS_SOURCE", 0.0)
    RHO_S = ffloat("RHO_S", 2500.0)
    RHO_C = None
    for key in ("RHO_CARRIER","RHO_FLUID","RHO_L","RHO_G","RHO_AIR","RHO0"):
        m2 = re.search(rf"{key}\s*=\s*([0-9\.\+EeDd-]+)", txt)
        if m2:
            RHO_C = f2float(m2.group(1)); break
    if RHO_C is None: RHO_C = 1.2
    G = f2float(re.search(r"GRA\s*=\s*([0-9\.\+EeDd-]+)", txt).group(1)) if re.search(r"GRA\s*=\s*([0-9\.\+EeDd-]+)", txt) else 9.81
    # time axis fallback
    T_START = ffloat("T_START", 0.0)
    DT_OUTPUT = ffloat("DT_OUTPUT", 1.0)
    return dict(PRES=PRES, X0=X0, Y0=Y0, DX=DX, NX=NX, NY=NY,
                XS=XS, YS=YS, t2=t2,
                PPF=PPF, ALPHAS=ALPHAS, RHO_S=RHO_S, RHO_C=RHO_C, G=G,
                T_START=T_START, DT_OUTPUT=DT_OUTPUT)

def nc_read_array(nc_path: Path, var_hint: str=None):
    # returns (name, data[nt,ny,nx], time[nt] or None)
    try:
        from netCDF4 import Dataset
        ds = Dataset(nc_path.as_posix(), "r")
        names = list(ds.variables.keys())
        cands = [var_hint] if var_hint else [n for n in names if ("pore" in n.lower() and "pres" in n.lower()) or ("porepres" in n.lower())]
        if not var_hint and not cands:
            cands = [n for n in names if "pres" in n.lower()]
        chosen = None
        for nm in cands:
            if nm and nm in ds.variables:
                v = ds.variables[nm]
                if hasattr(v, "dimensions") and any("time" in d.lower() for d in v.dimensions):
                    chosen = nm; break
        if chosen is None:
            for nm in cands:
                if nm and nm in ds.variables: chosen = nm; break
        if chosen is None:
            raise RuntimeError("No pressure-like variable found.")
        V = ds.variables[chosen][:]
        t = None
        if "time" in ds.variables: t = np.array(ds.variables["time"][:], dtype=float)
        ds.close()
        arr = np.array(V, dtype=float)
        if arr.ndim == 2: arr = arr[None, ...]
        return chosen, arr, t
    except Exception:
        import h5py
        f = h5py.File(nc_path.as_posix(), "r")
        def all_dsets(h):
            for k, v in h.items():
                if isinstance(v, h5py.Dataset): yield v.name, v
                elif isinstance(v, h5py.Group): yield from all_dsets(v)
        target = None; chosen = None
        if var_hint:
            for name, dset in all_dsets(f):
                if name.split("/")[-1] == var_hint:
                    target = dset; chosen = var_hint; break
        if target is None:
            for name, dset in all_dsets(f):
                low = name.lower()
                if ("pore" in low and "pres" in low) or ("porepres" in low):
                    target = dset; chosen = name.split("/")[-1]; break
        if target is None:
            for name, dset in all_dsets(f):
                if "pres" in name.lower():
                    target = dset; chosen = name.split("/")[-1]; break
        if target is None: raise SystemExit("No pressure-like dataset found (HDF5).")
        data = np.array(target[...], dtype=float)
        t = None
        if "time" in f.keys():
            try: t = np.array(f["time"][...], dtype=float).reshape(-1)
            except Exception: t=None
        f.close()
        if data.ndim == 2: data = data[None, ...]
        return chosen, data, t

def nc_read_h(nc_path: Path, var_hint: str=None):
    try:
        from netCDF4 import Dataset
        ds = Dataset(nc_path.as_posix(), "r")
        names = list(ds.variables.keys())
        cands = [var_hint] if var_hint else [n for n in names if n.lower() in ("h","thickness","flow_thickness","depth")]
        if not var_hint:
            cands += [n for n in names if "thick" in n.lower() or n.lower().endswith("_h")]
        chosen=None
        for nm in cands:
            if nm and nm in ds.variables:
                v = ds.variables[nm]
                if hasattr(v, "dimensions") and any("time" in d.lower() for d in v.dimensions):
                    chosen=nm; break
        if chosen is None:
            for nm in cands:
                if nm and nm in ds.variables: chosen=nm; break
        if chosen is None:
            ds.close(); return None, None
        V = ds.variables[chosen][:]
        ds.close()
        arr = np.array(V, dtype=float)
        if arr.ndim == 2: arr = arr[None, ...]
        return chosen, arr
    except Exception:
        try:
            import h5py
            f = h5py.File(nc_path.as_posix(), "r")
            def all_dsets(h):
                for k, v in h.items():
                    if isinstance(v, h5py.Dataset): yield v.name, v
                    elif isinstance(v, h5py.Group): yield from all_dsets(v)
            target=None; chosen=None
            if var_hint:
                for name,dset in all_dsets(f):
                    if name.split("/")[-1]==var_hint: target=dset; chosen=var_hint; break
            if target is None:
                for name,dset in all_dsets(f):
                    low=name.lower()
                    if low in ("h","thickness","flow_thickness","depth") or "thick" in low or low.endswith("_h"):
                        target=dset; chosen=name.split("/")[-1]; break
            if target is None: f.close(); return None, None
            data=np.array(target[...], dtype=float)
            f.close()
            if data.ndim==2: data=data[None,...]
            return chosen, data
        except Exception:
            return None, None

def pick_2x2_indices(X0, Y0, DX, XS, YS, NX, NY):
    # center-origin mapping: center(j) = X0 + j*DX ; center(i) = Y0 + i*DX
    j = int(round((XS - X0)/DX))
    i = int(round((YS - Y0)/DX))
    j0 = min(max(j, 0), NX-2)
    i0 = min(max(i, 0), NY-2)
    return i0, j0

def main():
    if len(sys.argv) < 2:
        print("Usage: excess_pp_compare_v3.py file.nc [--D 0.01] [--rho_mix 1600] [--varp VARNAME] [--varh VARNAME]")
        sys.exit(1)
    nc_path = Path(sys.argv[1])
    if not nc_path.exists():
        print(f"NC file not found: {nc_path}", file=sys.stderr); sys.exit(2)
    # optional args
    D = None; varp = VAR_HINT_PRESS; varh = VAR_HINT_H; rho_mix_override = RHO_MIX_OVERRIDE
    args = sys.argv[2:]; i=0
    while i < len(args):
        if args[i] == "--D" and i+1 < len(args):
            D = float(args[i+1]); i += 2; continue
        if args[i] == "--rho_mix" and i+1 < len(args):
            rho_mix_override = float(args[i+1]); i += 2; continue
        if args[i] == "--varp" and i+1 < len(args):
            varp = args[i+1]; i += 2; continue
        if args[i] == "--varh" and i+1 < len(args):
            varh = args[i+1]; i += 2; continue
        i += 1

    # INP
    inp = Path("IMEX_SfloW2D.inp")
    if not inp.exists():
        print("IMEX_SfloW2D.inp not found in cwd.", file=sys.stderr); sys.exit(3)
    I = parse_inp(inp)
    X0=I["X0"]; Y0=I["Y0"]; DX=I["DX"]; NX=I["NX"]; NY=I["NY"]
    XS=I["XS"]; YS=I["YS"]; t2=I["t2"]
    PRES=I["PRES"]; PPF=I["PPF"]; G=I["G"]
    if rho_mix_override is not None:
        rho_mix = rho_mix_override
    else:
        rho_mix = I["ALPHAS"]*I["RHO_S"] + (1.0 - I["ALPHAS"])*I["RHO_C"]
    if None in (X0,Y0,DX,XS,YS) or NX<=1 or NY<=1:
        print("Missing/invalid grid or source info in INP.", file=sys.stderr); sys.exit(4)

    # NC: pore pressure and thickness
    nameP, P3, time = nc_read_array(nc_path, varp)
    nt, ny, nx = P3.shape
    nameH, H3 = nc_read_h(nc_path, varh)
    if H3 is None:
        print("No thickness variable found in NC; need --varh or dataset with thickness.", file=sys.stderr); sys.exit(5)

    if time is None or len(time)!=nt:
        time = I["T_START"] + np.arange(nt)*I["DT_OUTPUT"]
    t = np.array(time, dtype=float)
    if t2 is None:
        t2 = float(time[1]) if len(time)>1 else 0.0
    t2_val = float(t2)
    idx2 = int(np.argmin(np.abs(t - t2_val)))

    # pick 2x2 around source
    i0,j0 = pick_2x2_indices(X0,Y0,DX,XS,YS,nx,ny)
    # series
    patchP = P3[:, i0:i0+2, j0:j0+2].reshape(nt, -1).mean(axis=1)
    excess_meas = patchP - PRES
    patchH = H3[:, i0:i0+2, j0:j0+2].reshape(nt, -1).mean(axis=1)

    # expected: pre-t2 uses h(t), post-t2 exponential decay from t2
    if D is None: D = D_DEFAULT
    h2 = float(patchH[idx2]) if np.isfinite(patchH[idx2]) and patchH[idx2] > 0 else max(np.nanmean(patchH[:idx2+1]), 1e-6)
    lam = (np.pi/2)**2 * D / (h2*h2)
    pre = rho_mix * G * PPF * patchH
    excess_pred = pre.copy()
    plateau_at_t2 = float(pre[idx2])
    mask = t >= t2_val
    excess_pred[mask] = plateau_at_t2 * np.exp(-lam*(t[mask]-t2_val))

    # --- main comparison plot ---
    fig = plt.figure(figsize=(10,6))
    plt.plot(t, excess_meas, label="Excess pore pressure (2×2 near source)")
    plt.plot(t, excess_pred, "--", label=f"Expected: ρ_mix g f h(t) up to t2, then exp decay\nρ_mix={rho_mix:.1f}, g={G:.3f}, f={PPF:.3f}, D={D:.3e}, h(t2)={h2:.3f}")
    plt.axvline(t2_val, alpha=0.3, label="t2 (source stop)")
    plt.xlabel("time (s)"); plt.ylabel("excess pore pressure (Pa)")
    plt.title(f"{nc_path.name}  @ source patch (varP={nameP}, varH={nameH})")
    plt.grid(alpha=0.3); plt.legend()
    out_png = Path(f"{nc_path.stem}_excess_compare_v3.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    # --- parameter plots (separate charts per guideline) ---
    # h(t)
    fig_h = plt.figure(figsize=(10,4))
    plt.plot(t, patchH, label="h(t) on 2×2 source patch")
    plt.xlabel("time (s)"); plt.ylabel("h (m)")
    plt.title("Thickness h(t) on source patch")
    plt.grid(alpha=0.3); plt.legend()
    out_h = Path(f"{nc_path.stem}_h_timeseries.png")
    plt.tight_layout(); plt.savefig(out_h, dpi=150); plt.close()

    # constants vs time (rho_mix, g, PPF) drawn as lines across t
    rho_series = np.full_like(t, rho_mix, dtype=float)
    g_series   = np.full_like(t, G, dtype=float)
    f_series   = np.full_like(t, PPF, dtype=float)

    fig_c = plt.figure(figsize=(10,4))
    plt.plot(t, rho_series, label="rho_mix (kg/m^3)")
    plt.plot(t, g_series,   label="g (m/s^2)")
    plt.plot(t, f_series,   label="PORE_PRES_FRACT (-)")
    plt.xlabel("time (s)"); plt.ylabel("value")
    plt.title("Constants over time on same axis")
    plt.grid(alpha=0.3); plt.legend()
    out_consts = Path(f"{nc_path.stem}_constants_timeseries.png")
    plt.tight_layout(); plt.savefig(out_consts, dpi=150); plt.close()

    # --- CSV dump including h, rho_mix, g, PPF ---
    out_csv = Path(f"{nc_path.stem}_excess_compare_v3.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("time_s,excess_meas_Pa,excess_pred_Pa,h_m,rho_mix_kgm3,g_mps2,PPF\n")
        for ti, em, ep, hh in zip(t, excess_meas, excess_pred, patchH):
            f.write(f"{ti:.12g},{em:.12g},{ep:.12g},{hh:.12g},{rho_mix:.12g},{G:.12g},{PPF:.12g}\n")

    print(f"Saved: {out_png.resolve()}")
    print(f"Saved: {out_h.resolve()}")
    print(f"Saved: {out_consts.resolve()}")
    print(f"Saved: {out_csv.resolve()}")
    print(f"[info] Cells used: i0={i0}, j0={j0}; t2={t2_val:.3f}s; rho_mix={rho_mix:.1f} kg/m^3; D={D:.3e} m^2/s")

if __name__ == "__main__":
    main()
