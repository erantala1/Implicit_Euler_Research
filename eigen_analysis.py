import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--prefix', default='KS_pred_Implicit_Euler_step_FNO_jacs_for_lead_100')
ap.add_argument('--fps', type=int, default=16)
args = ap.parse_args()

folder  = Path(args.folder).expanduser()
pattern = folder / f'{args.prefix}_chunk_*.npy'
files   = sorted(
    glob.glob(str(pattern)),
    key=lambda s: int(s.split('_chunk_')[-1].split('.npy')[0])
)
if not files:
    raise FileNotFoundError('no chunk files')

def align_eigpairs(ev, V):
    idx = np.argsort(ev.real)
    ev  = ev[idx]
    V   = V[:, idx]
    V   = V / np.sign(V[0:1, :]) 
    return ev, V

eig_vals = []
drift_blocks = []
prev_tail = None

for f in files:
    blk = np.load(f, allow_pickle=True).item()
    ev = blk['Eigenvalues'] 
    V  = blk['Eigenvectors']

    for m in range(ev.shape[0]):
        for j in range(ev.shape[1]):
            ev[m, j], V[m, j] = align_eigpairs(ev[m, j], V[m, j])

    eig_vals.append(ev) 

    inner = np.linalg.norm(V[1:] - V[:-1], axis=(2,3))
    drift_blocks.append(inner)

    if prev_tail is not None:
        cross = np.linalg.norm(V[0] - prev_tail, axis=(1,2))
        drift_blocks.append(cross[None, :])

    prev_tail = V[-1]
    del V, blk

E = np.concatenate(eig_vals, axis=0)
K, J, d = E.shape
print(f'\nEigen-values assembled:  {E.shape}')

drift = np.concatenate(drift_blocks, axis=0)
mean_drift = drift.mean(axis=1) 
print('drift array shape', drift.shape)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_aspect('equal', 'box')
ax.ticklabel_format(style="sci", scilimits=(-2,2))
colors = plt.cm.viridis(np.linspace(0,1,J))
scatters = [ax.plot([], [], 'o', ms=4, color=colors[j])[0] for j in range(J)]

def set_window(ev):
    rmax = np.abs(ev - ev.mean()).max()
    half = 1.05 * max(rmax, 1e-10)
    c = ev.mean()
    ax.set_xlim(c.real-half, c.real+half)
    ax.set_ylim(c.imag-half, c.imag+half)
    ax.ticklabel_format(style='plain')  

def init():
    for s in scatters: s.set_data([],[])
    set_window(E[0,0])
    ax.set_title('λ  t=0 iter=0')
    return scatters

def update(frame):
    k, j = divmod(frame, J)
    ev = E[k, j]
    set_window(ev)
    ax.set_title(f'λ  t={k} iter={j}')
    for jj, sc in enumerate(scatters):
        sc.set_data(E[k,jj].real, E[k,jj].imag)
    return scatters

ani = animation.FuncAnimation(fig, update, frames=K*J,
                              init_func=init, blit=True,
                              interval=1000/args.fps)
ani.save('eig_animation_lead_100.mp4', dpi=120, fps=args.fps)
print('saved eig_animation_lead_100.mp4')

plt.figure(figsize=(7,3))
plt.plot(range(1,K), mean_drift)
plt.xlabel('time step k')
plt.ylabel('avg ‖Δvec‖_F')
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig('eigvec_drift_lead_100.png', dpi=120)
print('saved eigvec_drift_lead_100.png')
