#!/usr/bin/env python3
"""
Animate eigen-values and show eigen-vector drift.

Assumes each *_chunk_*.npy contains
    'Eigenvalues' : (chunk, J, d)  complex
    'Eigenvectors': (chunk, J, d, d)  complex   ← columns = eigen-vectors
"""

import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------- CLI
ap = argparse.ArgumentParser(description="Animate eigenvalues and vector drift")
ap.add_argument('folder', help='Directory that holds *_chunk_*.npy files')
ap.add_argument('--prefix', default='KS_pred_Implicit_Euler_step_FNO_jacs_for_lead_100')
ap.add_argument('--fps', type=int, default=6)
ap.add_argument('--save', default='eig_animation_lead_100.mp4',
                help='Filename for the MP4 animation')
args = ap.parse_args()

folder  = Path(args.folder).expanduser()
files   = sorted(glob.glob(str(folder / f'{args.prefix}_chunk_*.npy')))
if not files:
    raise FileNotFoundError("No chunk files found")

# ---------------------------------------------------------------- load
eig_vals, eig_vecs = [], []
for f in files:
    blk = np.load(f, allow_pickle=True).item()
    if blk['Eigenvalues'].shape[0] == 0:   # skip empty chunks
        continue
    eig_vals.append(blk['Eigenvalues'].astype(np.complex64))   # (c,J,d)
    eig_vecs.append(blk['Eigenvectors'].astype(np.complex64))  # (c,J,d,d)

E  = np.concatenate(eig_vals, axis=0)   # (K,J,d)
V  = np.concatenate(eig_vecs, axis=0)   # (K,J,d,d)
K, J, d = E.shape
print(f'Loaded  eigenvalues: {E.shape}  eigenvectors: {V.shape}')

# ---------------------------------------------------------------- eigen-value animation (auto-limits)
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_aspect('equal', 'box')
ax.ticklabel_format(style="sci", scilimits=(-2,2))
colors = plt.cm.viridis(np.linspace(0,1,J))
scatters = [ax.plot([], [], 'o', ms=4, color=colors[j])[0] for j in range(J)]

def set_window(ev_block):
    """auto-zoom to current cloud with 10% margin"""
    rmax = np.abs(ev_block - ev_block.mean()).max()
    rmax = max(rmax, 1e-8)            # avoid zero window
    half = rmax * 1.1
    cx, cy = ev_block.real.mean(), ev_block.imag.mean()
    ax.set_xlim(cx-half, cx+half)
    ax.set_ylim(cy-half, cy+half)

def init():
    for s in scatters: s.set_data([], [])
    set_window(E[0,0])
    ax.set_title('λ,  time 0  iter 0')
    return scatters

def update(frame):
    k = frame // J          # outer step
    j = frame %  J          # inner iteration
    ev = E[k,j]             # (d,)
    set_window(ev)
    ax.set_title(f'λ,  time {k}  iter {j}')
    for jj, sc in enumerate(scatters):
        sc.set_data(E[k,jj].real, E[k,jj].imag)
    return scatters

ani = animation.FuncAnimation(fig, update, frames=K*J,
                              init_func=init, blit=True,
                              interval=1000/args.fps)
print('saving animation …')
ani.save(args.save, fps=args.fps, dpi=120)
print('saved as', args.save)

# ---------------------------------------------------------------- eigen-vector drift plot
# Frobenius norm ‖V_{k+1} − V_k‖  averaged over iterations
drift = np.linalg.norm(V[1:] - V[:-1], axis=(2,3))   # (K-1, J)
mean_drift = drift.mean(axis=1)                      # (K-1,)

plt.figure(figsize=(7,3))
plt.plot(range(1,K), mean_drift, lw=1.5)
plt.xlabel('time step k')
plt.ylabel('‖Δ eigenvectors‖_F  (avg over iter)')
plt.title('Eigen-vector change per time step')
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig('eigvec_drift_lead_100.png', dpi=120)
print('saved eigen-vector drift plot → eigvec_drift.png')
