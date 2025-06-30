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

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.axvline(0, color='grey', lw=0.5)
ax.axhline(0, color='grey', lw=0.5)

scatters = [ax.plot([], [], 'o', ms=3, label=f'iter {j}')[0] for j in range(J)]
ax.legend(loc='upper right', fontsize=8)

real_max = float(np.abs(E.real).max())
print(real_max)
imag_max = float(np.abs(E.imag).max())
print(imag_max)
lim = max(real_max, imag_max)

margin   = 1.1                # 10 % padding
if lim < 1e-4:                # eigenvalues ≪ 1 → use a fixed small box
    lim = 1e-3
ax.set_xlim(-lim * margin,  lim * margin)
ax.set_ylim(-lim * margin,  lim * margin)

# nicer tick labels for small ranges
ax.ticklabel_format(style="sci", scilimits=(-2, 2))

def init():
    for s in scatters:
        s.set_markersize(6)
        s.set_data([], [])
    ax.set_title('Eigenvalues (time step = 0)')
    return scatters

def update(frame):
    eig_frame = E[frame]            # (J, d)
    ax.set_title(f'Eigenvalues (time step = {frame})')
    for j, sc in enumerate(scatters):
        ev = eig_frame[j]           # (d,)
        sc.set_data(ev.real, ev.imag)
    return scatters

ani = animation.FuncAnimation(fig, update, frames=K,
                              init_func=init, blit=True,
                              interval=1000 / args.fps)

print('Saving animation… (requires ffmpeg)')
ani.save('eig_animation.mp4', fps=args.fps, dpi=120)
print('Done  →  eig_animation.mp4')
