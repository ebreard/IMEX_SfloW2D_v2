#!/usr/bin/env python3
# Author: ECP_BREARD
import sys, re
import numpy as np
from pathlib import Path

if len(sys.argv) != 15:
    print("Please provide 14 arguments:\n"
          " nx_cells ny_cells x_min x_max y_min y_max "
          " bed_xmin bed_xmax bed_ymin bed_ymax H0 alfas T plot_flag")
    sys.exit(1)

def as_int(s): return int(s)
def as_float(s): return float(s)

nx_cells = as_int(sys.argv[1]); ny_cells = as_int(sys.argv[2])
x_min = as_float(sys.argv[3]); x_max = as_float(sys.argv[4])
y_min = as_float(sys.argv[5]); y_max = as_float(sys.argv[6])
bed_xmin = as_float(sys.argv[7]);  bed_xmax = as_float(sys.argv[8])
bed_ymin = as_float(sys.argv[9]);  bed_ymax = as_float(sys.argv[10])
H0 = as_float(sys.argv[11]); alfas = as_float(sys.argv[12]); T = as_float(sys.argv[13])
plot_flag = sys.argv[14].lower() == 'true'

n_solid = 1
rho_s = 2000.0
SP_HEAT_S = 1617.0
SP_HEAT_A = 998.0
SP_GAS_CONST_A = 287.051
PRES = 101300.0
grav = 9.81

rho_a = PRES / (T * SP_GAS_CONST_A)
rho_m = alfas * rho_s + (1.0 - alfas) * rho_a
xs = alfas * rho_s / rho_m
SP_HEAT_MIX = xs * SP_HEAT_S + (1.0 - xs) * SP_HEAT_A

nx_points = nx_cells + 1; ny_points = ny_cells + 1
dx = (x_max - x_min) / float(nx_cells)
dy = (y_max - y_min) / float(ny_cells)

x = np.linspace(x_min, x_max, nx_points)
y = np.linspace(y_min, y_max, ny_points)
x_cent = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, nx_cells)
y_cent = np.linspace(y_min + 0.5*dy, y_max - 0.5*dy, ny_cells)
Xc, Yc = np.meshgrid(x_cent, y_cent)

H = np.where((Xc >= bed_xmin) & (Xc <= bed_xmax) & (Yc >= bed_ymin) & (Yc <= bed_ymax), H0, 0.0)
U = np.zeros_like(H); V = np.zeros_like(H)

Z = np.zeros((ny_points, nx_points))
header = (
    f"ncols     {nx_points}\n"
    f"nrows    {ny_points}\n"
    f"xllcorner {x_min - 0.5*dx}\n"
    f"yllcorner {y_min - 0.5*dx}\n"   # IMPORTANT
    f"cellsize {dx}\n"
    "NODATA_value -9999\n"
)
with open('topography_dem.asc','w') as f:
    np.savetxt(f, Z, header=header, fmt='%1.12f', comments='')

init_file = 'example_2D_0000.q_2d'
for j in range(ny_cells):
    q0 = np.zeros((7 + n_solid, nx_cells))
    q0[0,:] = x_cent
    q0[1,:] = y_cent[j]
    q0[2,:] = rho_m * H[j,:]
    q0[3,:] = rho_m * H[j,:] * U[j,:]
    q0[4,:] = rho_m * H[j,:] * V[j,:]
    q0[5,:] = rho_m * H[j,:] * SP_HEAT_MIX * T
    q0[6,:] = rho_s * H[j,:] * alfas / n_solid
    P_excess = rho_m * grav * H[j,:]                # EXCESS pore pressure (Pa)
    q0[-1,:] = P_excess * q0[2,:]                 # conservative var: (P_excess) * (rho*h)

    mode = "w+" if j==0 else "a"
    with open(init_file, mode) as f:
        np.savetxt(f, q0.T, fmt='%19.12e')
    with open(init_file, "a") as f:
        f.write(" \n")

tmpl = Path('IMEX_SfloW2D.template').read_text(encoding='utf-8', errors='ignore')
filedata = tmpl
filedata = filedata.replace('runname', 'example2D')
filedata = filedata.replace('restartfile', init_file)
filedata = filedata.replace('x_min', str(x_min))
filedata = filedata.replace('y_min', str(y_min))
filedata = filedata.replace('nx_cells', str(nx_cells))
filedata = filedata.replace('ny_cells', str(ny_cells))
filedata = filedata.replace('dx', str(dx))
Path('IMEX_SfloW2D.inp').write_text(filedata, encoding='utf-8')

yllcorner = x_min - 0.5*dx  # typo guard: recompute properly
yllcorner = y_min - 0.5*dx
print(f"Sanity: Y0 (from INP) = {y_min:.12g}")
print(f"        yllcorner + 0.5*cellsize = {(y_min - 0.5*dx) + 0.5*dx:.12g}")
print("Expect equality. If not equal, IMEX will complain.")

if plot_flag:
    import matplotlib.pyplot as plt
    Xc_plot, Yc_plot = np.meshgrid(x_cent, y_cent)
    plt.figure(figsize=(6,5))
    plt.pcolormesh(Xc_plot, Yc_plot, H, shading='auto')
    plt.colorbar(label='H (m)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('Initial rectangular bed')
    plt.tight_layout()
    plt.savefig('initial_bed.png', dpi=150)
    plt.close()
