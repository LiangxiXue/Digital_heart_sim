"""
Digital Heart — FitzHugh-Nagumo 2D Cardiac Electrophysiology Simulator

Simulates action-potential propagation on a 2D tissue grid using the
FitzHugh-Nagumo reaction-diffusion model, then exports the results as
animated videos / GIFs for both normal and arrhythmia scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os

# ──────────────────────────────────────────────
# 1. Parameter container
# ──────────────────────────────────────────────

@dataclass
class FHNParams:
    """All tuneable knobs for a FitzHugh-Nagumo simulation run."""
    N: int = 100                # grid side length (N x N)
    D: float = 1.0              # diffusion coefficient
    epsilon: float = 0.08       # time-scale separation (recovery speed)
    beta: float = 0.7           # recovery offset
    gamma: float = 0.8          # recovery damping
    dt: float = 0.1             # time step
    dx: float = 1.0             # spatial step
    n_steps: int = 3000         # total integration steps
    stim_region: Tuple[slice, slice] = (slice(0, 5), slice(0, 5))
    stim_value: float = 1.5     # initial membrane-potential kick
    dead_zones: List[Tuple[slice, slice]] = field(default_factory=list)

    def copy(self, **overrides) -> "FHNParams":
        """Return a shallow copy with selected fields overridden."""
        import copy as _copy
        p = _copy.copy(self)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p


# ──────────────────────────────────────────────
# 2. Grid initialisation
# ──────────────────────────────────────────────

def init_grid(params: FHNParams) -> Tuple[np.ndarray, np.ndarray]:
    """Create zero-initialised v and w grids, then apply the stimulus."""
    v = np.zeros((params.N, params.N), dtype=np.float64)
    w = np.zeros_like(v)
    v[params.stim_region] = params.stim_value
    return v, w


# ──────────────────────────────────────────────
# 3. Laplacian (five-point stencil, Neumann BC)
# ──────────────────────────────────────────────

def laplacian(v: np.ndarray, dx: float) -> np.ndarray:
    """
    Discrete Laplacian via the standard five-point stencil:

        ∇²v ≈ (v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4·v[i,j]) / dx²

    Boundary treatment — Neumann (zero-flux):
        Ghost cells outside the domain are set equal to the nearest interior
        value, which is equivalent to ∂v/∂n = 0 at every boundary face.
        In practice we use np.pad(mode='edge') to create a one-cell halo,
        then compute the stencil on the interior without any special-case
        branches.  This avoids the periodic wrap-around artefacts that a
        naive np.roll approach would produce.

    Numerical stability note:
        For the forward-Euler scheme to remain stable the CFL-like condition
        D·dt/dx² < 0.25 (in 2-D) must hold.  With the default parameters
        D=1, dt=0.1, dx=1 this ratio is 0.1 — safely below the threshold.
    """
    v_pad = np.pad(v, pad_width=1, mode='edge')
    return (
        v_pad[2:, 1:-1] + v_pad[:-2, 1:-1] +
        v_pad[1:-1, 2:] + v_pad[1:-1, :-2] -
        4.0 * v
    ) / (dx * dx)


# ──────────────────────────────────────────────
# 4. Single Euler step
# ──────────────────────────────────────────────

def euler_step(
    v: np.ndarray,
    w: np.ndarray,
    params: FHNParams,
    I_stim: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance (v, w) by one time step dt using the forward-Euler method.

        dv/dt = v - v³/3 - w + D·∇²v + I_stim
        dw/dt = ε·(v + β - γ·w)
    """
    lap = laplacian(v, params.dx)
    dv = v - (v ** 3) / 3.0 - w + params.D * lap
    if I_stim is not None:
        dv += I_stim
    dw = params.epsilon * (v + params.beta - params.gamma * w)

    v_new = v + params.dt * dv
    w_new = w + params.dt * dw

    # Enforce dead (necrotic) zones: clamp both variables to zero
    for zone in params.dead_zones:
        v_new[zone] = 0.0
        w_new[zone] = 0.0

    return v_new, w_new


# ──────────────────────────────────────────────
# 5. Full simulation runner
# ──────────────────────────────────────────────

def run_simulation(
    params: FHNParams,
    save_every: int = 10,
    extra_stim_fn=None,
) -> List[np.ndarray]:
    """
    Integrate the FHN system for params.n_steps and return snapshots of v
    every *save_every* steps (used as animation frames).

    Parameters
    ----------
    extra_stim_fn : callable(step, v, w) -> np.ndarray | None
        Optional callback that returns a spatiotemporal stimulus array
        for the current time step.  Useful for S1-S2 pacing protocols.
    """
    v, w = init_grid(params)
    frames = [v.copy()]

    for step in range(1, params.n_steps + 1):
        I_stim = None
        if extra_stim_fn is not None:
            I_stim = extra_stim_fn(step, v, w)
        v, w = euler_step(v, w, params, I_stim)
        if step % save_every == 0:
            frames.append(v.copy())

    return frames


# ──────────────────────────────────────────────
# 6. Animation builder
# ──────────────────────────────────────────────

def make_animation(
    frames: List[np.ndarray],
    filename: str,
    title: str = "FitzHugh-Nagumo  —  Membrane Potential  v",
    cmap: str = "magma",
    fps: int = 30,
    vmin: float = -2.5,
    vmax: float = 2.5,
    dpi: int = 120,
):
    """Render *frames* into an .mp4 or .gif and save to *filename*."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color='white', fontsize=13, pad=10)

    im = ax.imshow(
        frames[0], cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation='bilinear', origin='upper',
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    time_text = ax.text(
        0.02, 0.95, '', transform=ax.transAxes,
        color='white', fontsize=10, verticalalignment='top',
    )

    def _update(i):
        im.set_data(frames[i])
        time_text.set_text(f"frame {i}/{len(frames)-1}")
        return im, time_text

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, blit=True,
    )

    ext = os.path.splitext(filename)[1].lower()
    if ext == '.gif':
        anim.save(filename, writer='pillow', fps=fps, dpi=dpi)
    else:
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)

    plt.close(fig)
    print(f"[OK] Saved animation -> {filename}  ({len(frames)} frames)")


# ──────────────────────────────────────────────
# 7. Scenario definitions
# ──────────────────────────────────────────────

def scenario_normal() -> Tuple[FHNParams, dict]:
    """Normal sinus-like propagation from the upper-left corner."""
    params = FHNParams(
        N=100, D=1.0, epsilon=0.08, beta=0.7, gamma=0.8,
        dt=0.1, dx=1.0, n_steps=3000,
        stim_region=(slice(0, 5), slice(0, 5)),
        stim_value=1.5,
    )
    return params, dict(filename="normal_heartbeat.gif", title="Normal Heartbeat")


def scenario_arrhythmia_spiral() -> Tuple[FHNParams, dict]:
    """
    Induce a spiral (re-entrant) wave — the hallmark of many cardiac
    arrhythmias.  Strategy: S1-S2 cross-field stimulation protocol.

    1.  S1 — stimulate the left edge (plane wave travelling rightward).
    2.  S2 — after the wave has passed halfway, stimulate the bottom-left
         quadrant.  The broken wavefront curls into a spiral.
    """
    params = FHNParams(
        N=100, D=1.0, epsilon=0.08, beta=0.7, gamma=0.8,
        dt=0.1, dx=1.0, n_steps=5000,
        stim_region=(slice(None), slice(0, 5)),   # S1: full left edge
        stim_value=1.5,
    )

    def s2_stim(step, v, w):
        """Deliver a cross-field S2 stimulus at the right moment."""
        if step == 600:
            I = np.zeros_like(v)
            I[50:, :50] = 1.5
            return I
        return None

    return params, dict(
        filename="arrhythmia_heartbeat.gif",
        title="Arrhythmia — Spiral Re-entry",
        extra_stim_fn=s2_stim,
    )


def scenario_arrhythmia_block() -> Tuple[FHNParams, dict]:
    """
    Conduction block caused by a necrotic (scar) region in the tissue.
    The dead zone forces the wave to detour, creating shadow regions
    and potential wavelet break-up.
    """
    dead = [
        (slice(30, 70), slice(45, 55)),   # vertical scar band
    ]
    params = FHNParams(
        N=100, D=0.6, epsilon=0.08, beta=0.7, gamma=0.8,
        dt=0.1, dx=1.0, n_steps=4000,
        stim_region=(slice(0, 5), slice(0, 100)),   # top-edge plane wave
        stim_value=1.5,
        dead_zones=dead,
    )
    return params, dict(
        filename="arrhythmia_block.gif",
        title="Arrhythmia — Conduction Block (scar)",
    )


# ──────────────────────────────────────────────
# 8. Main entry point
# ──────────────────────────────────────────────

def main():
    scenarios = [
        ("Normal Heartbeat", scenario_normal),
        ("Spiral Re-entry Arrhythmia", scenario_arrhythmia_spiral),
        ("Conduction Block Arrhythmia", scenario_arrhythmia_block),
    ]

    for label, builder in scenarios:
        print(f"\n{'='*50}")
        print(f"  Running: {label}")
        print(f"{'='*50}")

        params, anim_kwargs = builder()
        extra_stim_fn = anim_kwargs.pop("extra_stim_fn", None)
        filename = anim_kwargs.pop("filename")
        title = anim_kwargs.pop("title")

        frames = run_simulation(params, save_every=10, extra_stim_fn=extra_stim_fn)
        make_animation(frames, filename=filename, title=title, **anim_kwargs)


if __name__ == "__main__":
    main()
