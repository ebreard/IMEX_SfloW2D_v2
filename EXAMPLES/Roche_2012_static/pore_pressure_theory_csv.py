#!/usr/bin/env python3
# Author: ECP_BREARD
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_AIR = 287.051  # J/(kg K)

def _parse_sci(s):
    if s is None: return None
    ss = str(s).strip().replace('d','E').replace('D','E')
    return float(ss)

def read_inp(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    def get_float(key, default=None):
        m = re.search(rf"{key}\s*=\s*([0-9\.\+EeDd-]+)", txt)
        return float(m.group(1).replace("D","E")) if m else default
    PRES = get_float("PRES", 101300.0)
    nu = get_float("KINEMATIC_VISCOSITY", None)
    if nu is None:
        for key in ["AIR_KINEMATIC_VISCOSITY","CARRIER_KINEMATIC_VISCOSITY"]:
            v = get_float(key, None)
            if v is not None: nu = v; break
    if nu is None: nu = 1.48e-5
    T  = get_float("AMBIENT_TEMPERATURE", None)
    if T is None:
        for key in ["TEMPERATURE","T0","T_START","T"]:
            v = get_float(key, None)
            if v is not None and v>1.0: T=v; break
    if T is None: T=300.0
    k = get_float("HYDRAULIC_PERMEABILITY", None)
    if k is None:
        for key in ["PERMEABILITY","K_PERM"]:
            v=get_float(key,None)
            if v is not None: k=v; break
    if k is None: k=1.0e-11
    return dict(PRES=PRES, nu=nu, T=T, k=k)

def read_csv_auto(csv_path: Path):
    df = pd.read_csv(csv_path)
    time_candidates = [c for c in df.columns if str(c).lower() in ("time","t","time_s","seconds","sec")]
    if not time_candidates:
        for c in df.columns:
            try:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy()
                if np.isfinite(v).sum()>5:
                    dv = np.diff(v[np.isfinite(v)])
                    if np.nanmedian(dv)>0: time_candidates=[c]; break
            except Exception: pass
    if not time_candidates:
        raise SystemExit("Could not find a time column in the CSV.")
    time_col = time_candidates[0]
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    exclude = set([time_col])
    data_candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if c in exclude: continue
        if any(key in cl for key in ["mean_excess","excess","p_excess","excess_pa","excess_nc","excess_pressure"]):
            if "pred" in cl or "model" in cl: 
                continue
            data_candidates.append(c)
    if not data_candidates:
        for c in df.columns:
            if c in exclude: continue
            v = pd.to_numeric(df[c], errors="coerce").to_numpy()
            if np.isfinite(v).sum()>5:
                data_candidates.append(c); break
    if not data_candidates:
        raise SystemExit("Could not find an excess-pressure column in the CSV.")
    data_col = data_candidates[0]
    data = pd.to_numeric(df[data_col], errors="coerce").to_numpy()
    m = np.isfinite(time) & np.isfinite(data)
    time = time[m]; data = data[m]
    return time, data, time_col, data_col

def main():
    D_override = _parse_sci(sys.argv[1]) if len(sys.argv)>=2 and not Path(sys.argv[1]).exists() else None
    csv_path = None
    if len(sys.argv)>=2 and Path(sys.argv[1]).exists():
        csv_path = Path(sys.argv[1])
    elif len(sys.argv)>=3:
        csv_path = Path(sys.argv[2])
    if csv_path is None:
        for name in ("example2D_excess_timeseries.csv","excess_diffusion_fit.csv"):
            p = Path(name)
            if p.exists(): csv_path=p; break
    if csv_path is None or (not csv_path.exists()):
        raise SystemExit("CSV not found. Pass it as argument or place example2D_excess_timeseries.csv in cwd.")

    h = float(_parse_sci(sys.argv[3])) if len(sys.argv)>=4 else 0.4

    time, excess, time_col, data_col = read_csv_auto(csv_path)

    if D_override is None:
        inp = Path("IMEX_SfloW2D.inp")
        if inp.exists():
            I = read_inp(inp)
            PRES, nu, T, k = I["PRES"], I["nu"], I["T"], I["k"]
            rho_air = PRES/(R_AIR*T)
            mu = rho_air*nu
            phi0 = 0.4
            D = k*PRES/(mu*phi0)
            D_source="computed"
        else:
            D = 0.1
            D_source="default(0.1)"
    else:
        D = float(D_override); D_source="CLI"

    P0 = float(excess[0])
    lam = (np.pi/2)**2 * D / (h*h)
    p_first = P0 * np.exp(-lam * time)

    out_csv = Path("excess_diffusion_fit_from_csv.csv")
    with open(out_csv,"w",encoding="utf-8") as f:
        f.write("time_s,excess_csv_Pa,pred_firstterm_Pa\n")
        for t, a, b in zip(time, excess, p_first):
            f.write(f"{t:.12g},{a:.12g},{b:.12g}\n")

    plt.figure(figsize=(8,5))
    plt.plot(time, excess, label=f"CSV {data_col}")
    plt.plot(time, p_first, "--", label=f"First-term model (D from {D_source})")
    plt.xlabel("time (seconds)")
    plt.ylabel("excess pore pressure (Pa)")
    plt.title(f"First-term diffusion fit (CSV)\nD={D:.3e} m²/s ({D_source}), h={h:.2f} m, P0={P0:.0f} Pa")
    plt.grid(alpha=0.3); plt.legend()
    out_png = Path("excess_diffusion_fit_from_csv.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    print(f"CSV read from: {csv_path.name}  (time='{time_col}', data='{data_col}')")
    print(f"Saved: {out_png.resolve()}")
    print(f"Saved: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
