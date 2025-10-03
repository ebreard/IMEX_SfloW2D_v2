#!/usr/bin/env python3
# Author: ECP Breard

import re, sys
from pathlib import Path
import numpy as np

# -- user params --
P = {
    "ncols": 100,
    "nrows": 100,
    "cellsize": 10.0,
    "xllcorner": 715000.0,
    "yllcorner": 1586002.0,
    "base_elev": 2000,
    "hole_depth": 20.0,
    "inlet_diameter": 300.0,
    "template_inp": "IMEX_SfloW2D.template",
    "out_inp":      "IMEX_SfloW2D.inp",
    "dem_file":     "hole_dem.asc",
}

# exact keys in your template
K = {
    "TOPOGRAPHY_FILE": ["TOPOGRAPHY_FILE"],
    "X0": ["X0"],
    "Y0": ["Y0"],
    "COMP_CELLS_X": ["COMP_CELLS_X"],
    "COMP_CELLS_Y": ["COMP_CELLS_Y"],
    "CELL_SIZE": ["CELL_SIZE"],
    "DY": ["DY"],  # if present -> enforce square cells by DY=CELL_SIZE
}
SK = {
    "X_SOURCE": ["X_SOURCE"],
    "Y_SOURCE": ["Y_SOURCE"],
    "R_SOURCE": ["R_SOURCE"],
    "R2_SOURCE": ["R2_SOURCE"],
    "ANGLE_SOURCE": ["ANGLE_SOURCE"],
    "SOURCE_DIAM": ["SOURCE_DIAM"],
    "SOURCE_Z": ["SOURCE_Z"],
}

def D0(x, dec=1): return f"{float(x):.{dec}f}D0"
def _s(x): return f"\"{x}\""

def grid_centers(nx, ny, dx, xll, yll):
    y_top = yll + ny*dx
    xs = xll + (np.arange(nx)+0.5)*dx
    ys = y_top - (np.arange(ny)+0.5)*dx
    return np.meshgrid(xs, ys)

def make_dem_with_hole(nx, ny, dx, xll, yll, z0, depth, d_in):
    Xc, Yc = grid_centers(nx, ny, dx, xll, yll)
    ic, jc = ny//2, nx//2  # center cell so source sits on a cell center
    xc, yc = Xc[ic, jc], Yc[ic, jc]
    Z = np.full((ny, nx), float(z0))
    R = np.sqrt((Xc-xc)**2 + (Yc-yc)**2)
    Z[R <= (0.5*d_in)] = z0 - depth
    return Z, (ic, jc), (xc, yc)

def write_esri_ascii(path, Z, nx, ny, xll, yll, dx):
    with Path(path).open("w", newline="\n") as f:
        f.write(f"ncols        {nx}\n")
        f.write(f"nrows        {ny}\n")
        f.write(f"xllcorner    {xll:.6f}\n")
        f.write(f"yllcorner    {yll:.6f}\n")
        f.write(f"cellsize     {dx:.6f}\n")
        f.write("NODATA_value -9999\n")
        for i in range(ny):
            f.write(" ".join(f"{v:.6f}" for v in Z[i, :]) + "\n")

def replace_first(text, keys, val_repr):
    for key in keys:
        m = re.search(rf"(^|\s)({re.escape(key)})\s*=\s*([^,\n]*)", text, re.IGNORECASE)
        if m:
            start, k, _old = m.groups()
            return text[:m.start()] + f"{start}{k}={val_repr}" + text[m.end():], True
    return text, False

def update_inp_from_template(tpl_path, out_path, dem_name, nx, ny, dx, xll, yll, xc, yc, d_in, z_floor):
    # CENTER origin (satisfies IMEX guard: X0 >= xll+0.5*dx)
    X0, Y0 = xll + 0.5*dx, yll + 0.5*dx
    R = 0.5*d_in
    text = tpl_path.read_text(errors="ignore")

    text, _ = replace_first(text, K["TOPOGRAPHY_FILE"], _s(dem_name))
    text, _ = replace_first(text, K["X0"], D0(X0))
    text, _ = replace_first(text, K["Y0"], D0(Y0))
    text, _ = replace_first(text, K["COMP_CELLS_X"], f"{nx}")
    text, _ = replace_first(text, K["COMP_CELLS_Y"], f"{ny}")
    text, _ = replace_first(text, K["CELL_SIZE"], D0(dx))
    text, _ = replace_first(text, K["DY"], D0(dx))  # enforce square cells if DY exists

    text, _ = replace_first(text, SK["X_SOURCE"], D0(xc))
    text, _ = replace_first(text, SK["Y_SOURCE"], D0(yc))
    text, _ = replace_first(text, SK["R_SOURCE"], D0(R))
    text, _ = replace_first(text, SK["R2_SOURCE"], D0(R))
    text, _ = replace_first(text, SK["ANGLE_SOURCE"], D0(0.0))
    text, _ = replace_first(text, SK["SOURCE_DIAM"], f"{d_in:.6f}")
    text, _ = replace_first(text, SK["SOURCE_Z"], f"{z_floor:.6f}")

    out_path.write_text(text)

def main():
    nx, ny = int(P["ncols"]), int(P["nrows"])
    dx = float(P["cellsize"])
    xll, yll = float(P["xllcorner"]), float(P["yllcorner"])
    z0, depth = float(P["base_elev"]), float(P["hole_depth"])
    d_in = float(P["inlet_diameter"])

    here = Path(".").resolve()
    tpl = here / P["template_inp"]
    out_inp = here / P["out_inp"]
    dem_path = here / P["dem_file"]

    if not tpl.exists():
        print(f"Template not found: {tpl}", file=sys.stderr); sys.exit(1)

    Z, (ic, jc), (xc, yc) = make_dem_with_hole(nx, ny, dx, xll, yll, z0, depth, d_in)
    write_esri_ascii(dem_path, Z, nx, ny, xll, yll, dx)

    z_floor = z0 - depth
    update_inp_from_template(tpl, out_inp, dem_path.name, nx, ny, dx, xll, yll, xc, yc, d_in, z_floor)

    # Center-origin extent checks vs DEM edges
    x_right = xll + nx*dx; y_top = yll + ny*dx
    X0, Y0 = xll + 0.5*dx, yll + 0.5*dx
    left_edge  = X0 - 0.5*dx; right_edge = X0 + (nx - 0.5)*dx
    bot_edge   = Y0 - 0.5*dx; top_edge   = Y0 + (ny - 0.5)*dx
    if abs(left_edge - xll) > 1e-9 or abs(right_edge - x_right) > 1e-9 \
       or abs(bot_edge - yll) > 1e-9 or abs(top_edge - y_top) > 1e-9:
        print("FATAL: center-origin extents mismatch.", file=sys.stderr); sys.exit(2)

    # source strictly inside interior
    if not (xll + 0.5*dx <= xc <= x_right - 0.5*dx and yll + 0.5*dx <= yc <= y_top - 0.5*dx):
        print("FATAL: source not inside interior cells.", file=sys.stderr); sys.exit(3)

    print("OK: DEM+INP generated from template and consistent (square cells).")
    print(f"DEM: {dem_path.name}")
    print(f"INP: {out_inp.name}")

if __name__ == "__main__":
    main()
