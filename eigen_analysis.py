import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ap = argparse.ArgumentParser(
    description="""Animate eigenvalues (real vs imaginary) stored in *_chunk_*.npy files.""")
ap.add_argument('folder', help='Directory that contains the chunk files')
ap.add_argument('--prefix', default='KS_pred_Implicit_Euler_step_FNO_jacs_for_1k',
                help='Filename prefix before _chunk_<n>.npy')
ap.add_argument('--chunks', nargs='*', type=int,
                help='Chunk numbers to load (default: all in folder)')
ap.add_argument('--fps', type=int, default=6, help='Frames per second')
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
for f in files:
    block = np.load(f, allow_pickle=True).item()
    if 'Eigenvalues' not in block:
        raise KeyError(f"'Eigenvalues' key missing in {f}")
    eigs_list.append(block['Eigenvalues'])          # (num_iters, d)
    print(f'  {Path(f).name:<40}  {block["Eigenvalues"].shape}')

E = np.concatenate(eigs_list, axis=0)               # (K, J, d)
K, J, d = E.shape
print(f'\nTotal outer steps K={K},  inner iterations J={J},  state dim d={d}')

# -------------------------------------------------- Matplotlib animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_aspect('equal', 'box')
ax.ticklabel_format(style="sci", scilimits=(-2, 2))

# one scatter per inner iteration
colors   = plt.cm.viridis(np.linspace(0, 1, J))
scatters = [ax.plot([], [], 'o', ms=4, color=colors[j], label=f'iter {j}')[0]
            for j in range(J)]
ax.legend(loc='upper right', fontsize=8)

MAG = 1_000      # <── magnification factor: 1e3, 1e4, …
MIN_BOX = 1e-4   # <── floor so the box never collapses to zero

def _set_window(ev_block):
    """Center window on current cloud & zoom by MAG."""
    center_real = ev_block.real.mean()
    center_imag = ev_block.imag.mean()
    radius      = np.abs(ev_block - (center_real + 1j*center_imag)).max()
    half = max(radius * MAG, MIN_BOX)
    ax.set_xlim(center_real - half, center_real + half)
    ax.set_ylim(center_imag - half, center_imag + half)

def init():
    for s in scatters:
        s.set_data([], [])
    _set_window(E[0])                     # initial limits
    ax.set_title('Eigenvalues (k=0, iter=0)')
    return scatters

def update(frame):
    k = frame // J
    j = frame %  J
    eig_frame = E[k, j]                  # (d,)
    _set_window(eig_frame)               # zoom & center
    ax.set_title(f'Eigenvalues (time step {k}, iter {j})')

    for jj, sc in enumerate(scatters):
        ev = E[k, jj]                    # plot all iterations each frame
        sc.set_data(ev.real, ev.imag)
    return scatters

total_frames = K * J                     # every (k,j) pair is a frame
ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              init_func=init, blit=True,
                              interval=1000,   # 1 s per frame
                              repeat=False)

ani.save('eig_animation.mp4', fps=8, dpi=120)
print('Done  →  eig_animation.mp4')