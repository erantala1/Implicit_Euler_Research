import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ap = argparse.ArgumentParser(
    description="""Animate eigenvalues (real vs imaginary) stored in *_chunk_*.npy files.""")
ap.add_argument('folder', help='Directory that contains the chunk files')
ap.add_argument('--prefix', default='KS_pred_Implicit_Euler_step_FNO_jacs_for_lead_100',
                help='Filename prefix before _chunk_<n>.npy')
ap.add_argument('--chunks', nargs='*', type=int,
                help='Chunk numbers to load (default: all in folder)')
args = ap.parse_args()

folder  = Path(args.folder).expanduser()
pattern = str(folder / f'{args.prefix}_chunk_*.npy')
files   = sorted(glob.glob(pattern))

if args.chunks is not None:
    files = [f for f in files
             if int(f.split('_chunk_')[-1].split('.npy')[0]) in args.chunks]

if not files:
    raise FileNotFoundError(f'No chunk files found for pattern: {pattern}')

print(f'Loading {len(files)} chunk(s)…')
eigs_list = []
eigvecs_list = []
for f in files:
    block = np.load(f, allow_pickle=True).item()
    if 'Eigenvalues' not in block:
        raise KeyError(f"'Eigenvalues' key missing in {f}")
    eigs_list.append(block['Eigenvalues'])
    eigvecs_list.append(block['Eigenvectors'])
    print(f'  {Path(f).name:<40}  {block["Eigenvalues"].shape}')
    print(f'  {Path(f).name:<40}  {block["Eigenvectors"].shape}')

E = np.concatenate(eigs_list, axis=0) 
V = np.concatenate(eigvecs_list,axis=0)
K, J, d = E.shape
print(f'\nTotal outer steps K={K},  inner iterations J={J},  state dim d={d}')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_aspect('equal', 'box')
ax.ticklabel_format(style="sci", scilimits=(-2,2))
colors = plt.cm.viridis(np.linspace(0,1,J))
scatters = [ax.plot([], [], 'o', ms=4, color=colors[j])[0] for j in range(J)]

def set_window(ev_block):
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
ani.save('eig_animation_lead_100.mp4', fps=8, dpi=120)
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
plt.savefig('eigvec_diff_lead_100.png', dpi=120)
print('saved eigen-vector diff plot → eigvec_diff.png')