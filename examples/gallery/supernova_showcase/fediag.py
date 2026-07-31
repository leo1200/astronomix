import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import casa_orlando as CO
from _common import centered_radius

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=256)
a = ap.parse_args()
args = argparse.Namespace(profile='casa_1d_map150.npz', n=a.n, age=350.0, nsnap=2,
    gpus=1, cooling=False, limiter_alpha=None, clump_sigma=1.0, csm_sigma=0.4,
    shell=True, shell_radius=1.5, shell_density=20.0, pistons=False,
    composition=True, save_state=None, ic_only=True)
state, config, params, rv, cu, hd, meta = CO.build(args)
s = np.asarray(state); i0 = rv.passive_scalar_index
dx = 7.0/a.n; vol = dx**3
r,_,_,_ = centered_radius(hd, 7.0, a.n); r = np.asarray(r)
rho = s[rv.density_index]; C_ej = s[i0]; C_Fe = s[i0+1]

d1 = np.load('casa_1d_map150.npz')
r1 = np.asarray(d1['r']); rho1 = np.asarray(d1['rho'])
menc1 = np.cumsum(4*np.pi*r1**2*rho1*np.gradient(r1))

print(f"\n n={a.n} dx={dx:.4f} pc")
print(f"{'r<':>7} {'3D M(<r)':>10} {'1D M(<r)':>10} {'ratio':>7} {'3D Fe(<r)':>10}")
for rr in (0.3, 0.5725, 0.8, 0.95, 1.0, 1.01, 1.05, 1.284):
    m3 = float(np.sum(np.where(r < rr, rho, 0.0))*vol)
    i = int(np.searchsorted(r1, rr)); m1 = float(menc1[min(i, len(menc1)-1)])
    fe = float(np.sum(np.where(r < rr, C_Fe*rho*C_ej, 0.0))*vol)
    print(f"{rr:7.4f} {m3:10.4f} {m1:10.4f} {m3/max(m1,1e-9):7.2f} {fe:10.4f}")
print(f"\n total 3D ejecta {float(np.sum(C_ej*rho)*vol):.4f}   total 3D Fe {float(np.sum(C_Fe*rho*C_ej)*vol):.4f}")
