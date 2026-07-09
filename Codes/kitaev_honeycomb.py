"""
Kitaev honeycomb model — zero magnetic field (κ = 0).

Directly follows the notes (kitaevhoneycomb.tex):

  Bond vectors A→B (δ_x + δ_y + δ_z = 0, notes l. 226):
    δ_x = (√3/2, −1/2),   δ_y = (−√3/2, −1/2),   δ_z = (0, 1)

  Bloch Hamiltonian (notes, after l. 243):
    h(q) = [[ 0,       i f(q)  ],
            [−i f*(q),   0     ]]

  where  f(q) = Jx exp(i q·δ_x) + Jy exp(i q·δ_y) + Jz exp(i q·δ_z)   [l. 266]

  Dispersion:  ε(q) = ±|f(q)|   [l. 271]

  Phase structure (from gap condition f(q) = 0):
    B phase (gapless):  |Jz| < |Jx|+|Jy|  AND  |Jy| < |Jx|+|Jz|  AND  |Jx| < |Jy|+|Jz|
    Ax phase (gapped):  |Jx| > |Jy|+|Jz|
    Ay phase (gapped):  |Jy| > |Jx|+|Jz|
    Az phase (gapped):  |Jz| > |Jx|+|Jy|

Plots (only the figures used in the report / supplement):
  1. Phase diagram on the (Jx, Jy, Jz) simplex          -> HC_1_phase_diagram.pdf
  3. Dirac cone at K — 2-panel (BZ map + 3D zoom)        -> HC_3_dirac_cone.pdf
  5. Berry curvature Ω(q) and Chern number C             -> HC_5_chern_number.pdf
  6. Gap opening at the Dirac point (κ=0 vs κ≠0)         -> HC_6_gap_opening.pdf
  9. Zero-mode localization across the ribbon            -> HC_9_zero_modes.pdf

Usage:
    python kitaev_honeycomb.py        # all plots
    python kitaev_honeycomb.py 1 3    # plots 1 and 3 only
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

PLOTS = Path(__file__).parent.parent / "Note" / "Plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

_registry: dict[int, tuple] = {}


def plot(index: int, label: str = ""):
    def decorator(fn):
        _registry[index] = (label or fn.__name__, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class KitaevHoneycomb:
    """
    Kitaev honeycomb model, flux-free sector (w_p = +1).

    kappa = 0   : gapless B phase  (notes §3)
    kappa ≠ 0   : gapped B phase from weak magnetic field (notes §3 extension)
                  effective NNN coupling κ ~ h_x h_y h_z / (4 J_x J_y J_z)
    """

    # Nearest-neighbour bond vectors A→B (notes l. 226)
    DX = np.array([ np.sqrt(3) / 2, -0.5])
    DY = np.array([-np.sqrt(3) / 2, -0.5])
    DZ = np.array([0.0,              1.0 ])

    # NNN displacement vectors n_i = δ_α − δ_β  (A→A, counterclockwise)
    N1 = DX - DZ   # ( √3/2, −3/2)
    N2 = DY - DX   # (−√3,    0  )
    N3 = DZ - DY   # ( √3/2,  3/2)

    # Primitive Bravais vectors
    A1 = DZ - DX
    A2 = DZ - DY

    # Reciprocal lattice vectors  A_i · B_j = 2π δ_ij
    B1 = np.array([-2 * np.pi / np.sqrt(3),  2 * np.pi / 3])
    B2 = np.array([ 2 * np.pi / np.sqrt(3),  2 * np.pi / 3])

    def __init__(self, Jx=1.0, Jy=1.0, Jz=1.0, kappa=0.0):
        self.Jx    = Jx
        self.Jy    = Jy
        self.Jz    = Jz
        self.kappa = kappa

    # --- NN hopping amplitude -------------------------------------------------

    def f(self, qx, qy):
        """f(q) = Σ_α J_α exp(i q·δ_α)   [notes l. 266]. Gap closes at f=0."""
        return (
            self.Jx * np.exp(1j * (self.DX[0]*qx + self.DX[1]*qy))
            + self.Jy * np.exp(1j * (self.DY[0]*qx + self.DY[1]*qy))
            + self.Jz * np.exp(1j * (self.DZ[0]*qx + self.DZ[1]*qy))
        )

    # --- NNN magnetic mass (κ ≠ 0) -------------------------------------------

    def mass(self, qx, qy):
        """
        NNN mass m(q) = 2κ Σ_i sin(q·n_i).
        d_z component of the Bloch Hamiltonian opened by the magnetic field.
        At K: m(K) = −3√3 κ  →  gap Δ = 6√3 |κ|.
        """
        return 2.0 * self.kappa * (
            np.sin(self.N1[0]*qx + self.N1[1]*qy)
            + np.sin(self.N2[0]*qx + self.N2[1]*qy)
            + np.sin(self.N3[0]*qx + self.N3[1]*qy)
        )

    # --- Dispersion -----------------------------------------------------------

    def dispersion(self, qx, qy):
        """ε(q) = √(|f|² + m²). Reduces to |f| when κ=0."""
        fq = self.f(qx, qy)
        mq = self.mass(qx, qy)
        return np.sqrt(np.abs(fq)**2 + mq**2)

    # --- d-vector and Berry curvature (κ ≠ 0) --------------------------------

    def dvec(self, qx, qy):
        """
        d-vector: d = (−Re f, −Im f, m),  shape (3, *shape(qx)).
        h(q) = d·σ = d_x σ_x + d_y σ_y + d_z σ_z.
        """
        fq = self.f(qx, qy)
        mq = np.asarray(self.mass(qx, qy), dtype=float)
        return np.array([-fq.real, -fq.imag, mq])

    def berry_curvature(self, qx, qy, dq: float = 1e-4):
        """
        Berry curvature of the lower band via the d-vector formula:
          Ω(q) = −1/(2|d|³) d·(∂_qx d × ∂_qy d).
        Finite differences with step dq.
        """
        d   = self.dvec(qx, qy)
        ddx = (self.dvec(qx + dq, qy) - self.dvec(qx - dq, qy)) / (2*dq)
        ddy = (self.dvec(qx, qy + dq) - self.dvec(qx, qy - dq)) / (2*dq)
        # cross product along vector axis 0
        cross = np.cross(ddx, ddy, axisa=0, axisb=0, axisc=0)
        E   = np.sqrt(np.einsum('i...,i...->...', d, d))   # |d|
        dot = np.einsum('i...,i...->...', d, cross)         # d·cross
        return -dot / (2.0 * E**3)

    def chern_number(self, n: int = 60) -> int:
        """
        First Chern number of the lower band via the
        Fukui–Hatsugai–Suzuki (FHS) lattice method.

        Discretises the BZ as q(i,j) = (i/n)B1 + (j/n)B2 (periodic),
        builds the U(1) link variables, and sums the plaquette curvature.
        Returns an integer; requires a gapped spectrum (κ ≠ 0 in B phase).
        """
        b1, b2 = self.B1, self.B2
        II, JJ = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

        # q-points on the BZ grid
        QXg = (II/n)*b1[0] + (JJ/n)*b2[0]   # (n, n)
        QYg = (II/n)*b1[1] + (JJ/n)*b2[1]

        # Batch-diagonalise all 2×2 Hamiltonians at once
        fq  = self.f(QXg, QYg)               # (n, n) complex
        mq  = self.mass(QXg, QYg)            # (n, n) float
        H   = np.zeros((n, n, 2, 2), dtype=complex)
        H[:, :, 0, 0] =  mq
        H[:, :, 1, 1] = -mq
        H[:, :, 0, 1] =  1j * fq
        H[:, :, 1, 0] = -1j * np.conj(fq)
        _, vecs = np.linalg.eigh(H)           # (n, n, 2, 2), sorted ascending
        u = vecs[:, :, :, 0]                  # lower eigenvector, (n, n, 2)

        # Periodic rolls for neighbours
        u_ix  = np.roll(u, -1, axis=0)        # u[i+1, j]
        u_jx  = np.roll(u, -1, axis=1)        # u[i, j+1]
        u_ijx = np.roll(u_ix, -1, axis=1)     # u[i+1, j+1]

        # U(1) link variables (inner products)
        Ux  = np.einsum('ijk,ijk->ij', u.conj(), u_ix)
        Uy  = np.einsum('ijk,ijk->ij', u.conj(), u_jx)
        Uxp = np.einsum('ijk,ijk->ij', u_jx.conj(), u_ijx)
        Uyp = np.einsum('ijk,ijk->ij', u_ix.conj(), u_ijx)

        Ux  /= np.abs(Ux);  Uy  /= np.abs(Uy)
        Uxp /= np.abs(Uxp); Uyp /= np.abs(Uyp)

        # Plaquette: F = Ux · Uyp · Uxp* · Uy*
        F = Ux * Uyp * np.conj(Uxp) * np.conj(Uy)
        return int(np.round(np.sum(np.angle(F)) / (2*np.pi)))

    # --- Ribbon (strip) Hamiltonian for edge modes ---------------------------

    def ribbon_hamiltonian(self, k, M):
        """
        Bloch Hamiltonian H(k) of a honeycomb strip: M unit cells stacked along
        A1 (open, finite width) and periodic along t1 = A2 − A1 = (√3, 0).

        Dimensionless k ∈ [−π, π] is conjugate to the cell index along t1; the
        two bulk Dirac points project to k = ±2π/3. Basis order per row m is
        [A_m, B_m], so H has size 2M.

        The matrix is the Fourier transform (along t1 only) of the real,
        antisymmetric Majorana coupling: the NN part reproduces h_AB = i f(q),
        and the NNN Haldane part reproduces the mass d_z = m(q) (amplitude −iκ
        on A→A, +iκ on B→B, along the counter-clockwise bonds +n_i).

        Physics:
          κ = 0  → chiral (sublattice) symmetry S H S = −H protects an E = 0
                   edge flat band spanning |k| < 2π/3 (between the Dirac points).
          κ ≠ 0  → the mass gaps the bulk (Δ = 6√3|κ|) and disperses the flat
                   band into ONE chiral Majorana edge mode per edge, crossing
                   E = 0 at k = 0 — the |C| = 1 bulk–boundary correspondence.
        """
        Jx, Jy, Jz, kap = self.Jx, self.Jy, self.Jz, self.kappa
        H = np.zeros((2 * M, 2 * M), dtype=complex)
        A = lambda m: 2 * m          # A-site index in row m
        B = lambda m: 2 * m + 1      # B-site index in row m

        def add(a, b, val):          # add a hopping + its Hermitian conjugate
            H[a, b] += val
            H[b, a] += np.conj(val)

        ek, emk = np.exp(1j * k), np.exp(-1j * k)
        for m in range(M):
            # NN  A_m → B  (amplitude i J_α):  z-bond same cell, x/y-bonds in row m−1
            add(A(m), B(m), 1j * Jz)                       # z : (Δm,Δn)=( 0, 0)
            if m - 1 >= 0:
                add(A(m), B(m - 1), 1j * Jx)               # x : (Δm,Δn)=(−1, 0)
                add(A(m), B(m - 1), 1j * Jy * emk)         # y : (Δm,Δn)=(−1,−1)
            # NNN Haldane mass along +n_i:  A→A amplitude −iκ,  B→B amplitude +iκ
            add(A(m), A(m), -1j * kap * emk)               # n2 : (0,−1)
            add(B(m), B(m), +1j * kap * emk)
            if m - 1 >= 0:
                add(A(m), A(m - 1), -1j * kap)             # n1 : (−1, 0)
                add(B(m), B(m - 1), +1j * kap)
            if m + 1 < M:
                add(A(m), A(m + 1), -1j * kap * ek)        # n3 : (+1,+1)
                add(B(m), B(m + 1), +1j * kap * ek)
        return H

    # --- Static helpers -------------------------------------------------------

    @staticmethod
    def dirac_points():
        """K and K' at the BZ corners (isotropic case)."""
        K  = np.array([2*np.pi / (3*np.sqrt(3)),  2*np.pi / 3])
        Kp = np.array([4*np.pi / (3*np.sqrt(3)),  0.0         ])
        return K, Kp

    @classmethod
    def bz_hexagon(cls):
        """Six BZ corners (closed, 7 points) at radius r = |B1|/√3."""
        r = np.linalg.norm(cls.B1) / np.sqrt(3)
        angles = np.arange(7) * np.pi / 3
        return r * np.cos(angles), r * np.sin(angles)

    @classmethod
    def bz_grid(cls, n=200):
        """Meshgrid over the first BZ. Returns (QX, QY, mask)."""
        extent = np.linalg.norm(cls.B1) * 0.72
        qx = np.linspace(-extent, extent, n)
        qy = np.linspace(-extent, extent, n)
        QX, QY = np.meshgrid(qx, qy)
        B3 = cls.B1 + cls.B2
        def hs(B): return float(np.dot(B, B)) / 2.0
        mask = (
            (np.abs(QX*cls.B1[0] + QY*cls.B1[1]) <= hs(cls.B1))
            & (np.abs(QX*cls.B2[0] + QY*cls.B2[1]) <= hs(cls.B2))
            & (np.abs(QX*B3[0]    + QY*B3[1]    ) <= hs(B3))
        )
        return QX, QY, mask

    @staticmethod
    def bulk_gap(Jx, Jy, Jz):
        """Analytical bulk gap (κ=0): 0 in B phase, Δ=max_J−(sum of others) in A."""
        J = sorted([Jx, Jy, Jz])
        return max(0.0, J[2] - J[1] - J[0])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bz_overlay(ax):
    """Draw the hexagonal BZ boundary + label K, K' on a 2-D axes."""
    bx, by = KitaevHoneycomb.bz_hexagon()
    ax.plot(bx, by, color="white", lw=1.8, zorder=5)

    K, Kp = KitaevHoneycomb.dirac_points()
    r = float(np.linalg.norm(K))
    for i, ang in enumerate(np.arange(6) * np.pi / 3):
        pt = np.array([r*np.cos(ang), r*np.sin(ang)])
        col = "yellow" if i % 2 == 1 else "cyan"
        ax.scatter(*pt, color=col, s=70, zorder=8)

    off = 0.18
    ax.annotate(r"$\mathbf{K}$",  K,  xytext=K  + [ off,  off], fontsize=10,
                color="yellow", fontweight="bold")
    ax.annotate(r"$\mathbf{K'}$", Kp, xytext=Kp + [ off, -off - 0.12], fontsize=10,
                color="cyan",   fontweight="bold")


# ---------------------------------------------------------------------------
# Plot 1 — Phase diagram on the (Jx, Jy, Jz) simplex
# ---------------------------------------------------------------------------

@plot(1, "Phase diagram on the (Jx,Jy,Jz) simplex")
def plot_phase_diagram(N=420):
    """
    Ternary phase diagram with Jx+Jy+Jz = 1, Jx,Jy,Jz ≥ 0.

    Phase boundaries (from gap condition f(q)=0):
      Jx = 1/2  (boundary between Ax and B/Ay/Az)
      Jy = 1/2
      Jz = 1/2
    These divide the simplex into 4 regions:
      B  (centre): all Ji < 1/2  — gapless, linear Dirac dispersion
      Ax (corner): Jx > 1/2     — gapped
      Ay (corner): Jy > 1/2     — gapped
      Az (corner): Jz > 1/2     — gapped

    Colour encodes the bulk gap Δ = max(0, max_J − (sum − max_J)).
    """
    # --- sample the simplex with a triangular grid --------------------------
    # barycentric coordinates: Jx = i/N, Jy = j/N, Jz = k/N, i+j+k = N
    ii = []
    jj = []
    kk = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            ii.append(i)
            jj.append(j)
            kk.append(k)
    Jx = np.array(ii, dtype=float) / N
    Jy = np.array(jj, dtype=float) / N
    Jz = np.array(kk, dtype=float) / N

    # Cartesian positions in the triangle
    # corner Jx=1 → (0,0), Jy=1 → (1,0), Jz=1 → (0.5, √3/2)
    x = Jy + 0.5 * Jz
    y = (np.sqrt(3) / 2) * Jz

    # Bulk gap at each point
    J_sorted = np.sort(np.column_stack([Jx, Jy, Jz]), axis=1)   # each row sorted
    gap = np.maximum(0.0, J_sorted[:, 2] - J_sorted[:, 1] - J_sorted[:, 0])

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    triang = mtri.Triangulation(x, y)
    tcf = ax.tricontourf(triang, gap, levels=40, cmap="Blues", vmin=0, vmax=0.5)
    ax.tricontour(triang, gap, levels=[0.0], colors="black", linewidths=1.8)
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"Bulk gap $\Delta$  (in units of $J_x+J_y+J_z$)", fontsize=9)

    # Phase boundaries (midpoints of triangle edges)
    M_xy = (0.50, 0.00)   # midpoint of Jx=1 — Jy=1 edge:  Jx=Jy=1/2, Jz=0
    M_xz = (0.25, np.sqrt(3)/4)  # midpoint of Jx=1 — Jz=1: Jx=Jz=1/2, Jy=0
    M_yz = (0.75, np.sqrt(3)/4)  # midpoint of Jy=1 — Jz=1: Jy=Jz=1/2, Jx=0
    for p, q_ in [(M_xy, M_xz), (M_xy, M_yz), (M_xz, M_yz)]:
        ax.plot([p[0], q_[0]], [p[1], q_[1]], "k-", lw=2.0, zorder=4)

    # Region centroids
    cx, cy  = 0.50, np.sqrt(3)/6          # B centroid = isotropic point (0.5, 0.289)
    Ax_c    = (0.22, np.sqrt(3)/12)        # Ax centroid ≈ (0.22, 0.144)
    Ay_c    = (0.78, np.sqrt(3)/12)        # Ay centroid ≈ (0.78, 0.144)
    Az_c    = (0.50, np.sqrt(3)/3)         # Az centroid ≈ (0.50, 0.577)

    # Phase labels — "B" shifted up so it doesn't sit on the isotropic dot
    ax.text(cx, cy + 0.07, "B\n(gapless)", ha="center", va="center",
            fontsize=12, fontweight="bold", color="navy")
    ax.text(*Ax_c, r"$A_x$", ha="center", va="center", fontsize=11, color="black")
    ax.text(*Ay_c, r"$A_y$", ha="center", va="center", fontsize=11, color="black")
    ax.text(*Az_c, r"$A_z$", ha="center", va="center", fontsize=11, color="black")

    # Outer triangle frame
    tri_patch = mpatches.Polygon(
        [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)],
        closed=True, fill=False, edgecolor="black", lw=2.2, zorder=5
    )
    ax.add_patch(tri_patch)

    # Corner labels
    ax.text(-0.04, -0.06, r"$J_x=1$", ha="center", fontsize=10)
    ax.text( 1.04, -0.06, r"$J_y=1$", ha="center", fontsize=10)
    ax.text( 0.58,  np.sqrt(3)/2 , r"$J_z=1$", ha="center", fontsize=10)

    # Isotropic point Jx=Jy=Jz=1/3 at the B centroid
    ax.scatter(cx, cy, color="red", s=55, zorder=7)
    ax.annotate(r"$J_x\!=\!J_y\!=\!J_z$", (cx, cy),
                xytext=(0.45, 0.13), fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.9))

    # Phase boundary labels — placed ON the inner phase boundaries
    # Jz=1/2: horizontal boundary at y=√3/4≈0.433, label just above it
    ax.text(0.50, np.sqrt(3)/4 + 0.025, r"$J_z\!=\!1/2$",
            ha="center", va="bottom", fontsize=8.5, color="black", rotation=0)
    # Jx=1/2: left diagonal (M_xz → M_xy), midpoint ≈ (0.375, 0.217)
    ax.text(0.29, 0.20, r"$J_x\!=\!1/2$",
            ha="center", va="center", fontsize=8.5, color="black", rotation=60)
    # Jy=1/2: right diagonal (M_xy → M_yz), midpoint ≈ (0.625, 0.217)
    ax.text(0.71, 0.20, r"$J_y\!=\!1/2$",
            ha="center", va="center", fontsize=8.5, color="black", rotation=-60)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Kitaev honeycomb: phase diagram\n"
        r"($J_x+J_y+J_z=1$;  blue shade = bulk gap $\Delta$)",
        pad=10
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "HC_1_phase_diagram.pdf")
    plt.close(fig)
    print("  [1] Saved  HC_1_phase_diagram.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Dirac cone at K (2-panel: BZ map + 3D zoom)
# ---------------------------------------------------------------------------

@plot(3, "Dirac cone at K — 2-panel")
def plot_dirac_cone(Jx=1.0, Jy=1.0, Jz=1.0, n_bz=200, n_cone=160, radius=0.01):
    """
    Left:  pcolormesh of |f(q)| in the first BZ.
           Dark spots at the 6 BZ corners show where |f(q)| = 0 (Dirac points).
           The red box marks the zoomed region shown on the right.
    Right: Linear dispersion ε(K+δq) ≈ v_F |δq| in the vicinity of K.
           radius = 0.18 << r_BZ ≈ 2.42 keeps us firmly in the linear regime.
           This is the strict Dirac cone: the apex at (0,0,0) is the K point.
    """
    model = KitaevHoneycomb(Jx, Jy, Jz)
    K, _ = KitaevHoneycomb.dirac_points()

    # BZ data
    QX, QY, mask = model.bz_grid(n_bz)
    F_abs = np.where(mask, model.dispersion(QX, QY), np.nan)

    # Cone data in relative coordinates (δq centred at 0)
    dq = np.linspace(-radius, radius, n_cone)
    DQX, DQY = np.meshgrid(dq, dq)
    E = model.dispersion(K[0] + DQX, K[1] + DQY)

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5.5), layout="constrained")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax2d = fig.add_subplot(gs[0])
    ax3d = fig.add_subplot(gs[1], projection="3d")

    # --- left: BZ map -------------------------------------------------------
    pcm = ax2d.pcolormesh(QX, QY, F_abs, cmap="inferno", shading="auto", vmin=0)
    fig.colorbar(pcm, ax=ax2d, label=r"$|f(\mathbf{q})|$", fraction=0.046, pad=0.04)
    _bz_overlay(ax2d)

    rect = mpatches.Rectangle(
        (K[0] - radius, K[1] - radius), 2*radius, 2*radius,
        linewidth=2, edgecolor="red", facecolor="none", zorder=10
    )
    ax2d.add_patch(rect)
    ax2d.text(K[0] + radius + 0.06, K[1], "⟶",
              fontsize=13, color="red", ha="left", va="center", clip_on=False)

    lim = float(np.nanmax(np.abs(QX))) * 1.02
    ax2d.set_xlim(-lim, lim)
    ax2d.set_ylim(-lim, lim)
    ax2d.set_aspect("equal")
    ax2d.set_xlabel(r"$q_x$")
    ax2d.set_ylabel(r"$q_y$")
    ax2d.set_title(
        r"$|f(\mathbf{q})|$ in first BZ"
    )

    # --- right: 3D Dirac cone in relative coordinates -----------------------
    surf_kw = dict(linewidth=0, antialiased=True, alpha=0.88)
    ax3d.plot_surface(DQX, DQY,  E, cmap="coolwarm", **surf_kw)
    ax3d.plot_surface(DQX, DQY, -E, cmap="coolwarm", **surf_kw)

    # Apex at the origin = the Dirac point K
    ax3d.scatter(0.0, 0.0, 0.0, color="yellow", s=100, depthshade=False, zorder=10)

    Emax = float(np.max(E))
    ax3d.set_xlabel(r"$\delta q_x$", labelpad=5)
    ax3d.set_ylabel(r"$\delta q_y$", labelpad=5)
    ax3d.set_zlabel(r"$\varepsilon$", labelpad=5)
    ax3d.set_zlim(-Emax * 1.05, Emax * 1.05)
    ax3d.set_title(
        r"Linear dispersion near $\mathbf{K}$:  $\varepsilon \approx v_F|\delta\mathbf{q}|$"
        "\n"
        r"($|\delta\mathbf{q}| \ll r_{\mathrm{BZ}}$, strictly linear regime)"
    )
    ax3d.view_init(elev=25, azim=225)

    fig.suptitle(
        r"Dirac cone at $\mathbf{K}$  ($J_x=J_y=J_z=1$)",
        fontsize=12
    )
    fig.savefig(PLOTS / "HC_3_dirac_cone.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [3] Saved  HC_3_dirac_cone.pdf")


# ---------------------------------------------------------------------------
# Plot 5 — Berry curvature Ω(q) and Chern number C
# ---------------------------------------------------------------------------

@plot(5, "Berry curvature Ω(q) and Chern number")
def plot_chern(Jx=1.0, Jy=1.0, Jz=1.0, kappa=0.1, n_bc=160, n_fhs=60):
    """
    Left:  2-D map of Berry curvature Ω(q) in the first BZ.
           Peaks at K and K' carry most of the integrated weight.
           FHS Chern number C is computed and annotated.
    Right: C vs κ — step function C = sign(κ) = ±1 for all κ ≠ 0
           (computed by the FHS method at several κ values).
    """
    model = KitaevHoneycomb(Jx, Jy, Jz, kappa=kappa)
    QX, QY, mask = model.bz_grid(n_bc)
    Omega = np.where(mask, model.berry_curvature(QX, QY), np.nan)

    # FHS Chern number for current κ
    C = model.chern_number(n_fhs)

    # C vs κ sweep
    kappa_sweep = [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20]
    C_sweep = [KitaevHoneycomb(Jx, Jy, Jz, kappa=k).chern_number(n_fhs)
               for k in kappa_sweep]

    fig = plt.figure(figsize=(13, 5.5), layout="constrained")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    axbc = fig.add_subplot(gs[0])
    axch = fig.add_subplot(gs[1])

    # --- left: Berry curvature map -------------------------------------------
    vmax = float(np.nanpercentile(np.abs(Omega), 98))
    pcm  = axbc.pcolormesh(QX, QY, Omega, cmap="RdBu_r", shading="auto",
                           vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=axbc, label=r"$\Omega(\mathbf{q})$", fraction=0.046, pad=0.04)

    bx, by = KitaevHoneycomb.bz_hexagon()
    axbc.plot(bx, by, color="black", lw=1.5)
    K, Kp = KitaevHoneycomb.dirac_points()
    axbc.scatter(*K,  color="yellow", s=60, zorder=6)
    axbc.scatter(*Kp, color="cyan",   s=60, zorder=6)
    axbc.annotate(r"$\mathbf{K}$",  K,  xytext=K  + 0.18, fontsize=9,
                  color="yellow", fontweight="bold")
    axbc.annotate(r"$\mathbf{K'}$", Kp, xytext=Kp + np.array([0.18, -0.20]),
                  fontsize=9, color="cyan", fontweight="bold")

    lim = float(np.nanmax(np.abs(QX)))*1.02
    axbc.set_xlim(-lim, lim); axbc.set_ylim(-lim, lim)
    axbc.set_aspect("equal")
    axbc.set_xlabel(r"$q_x$"); axbc.set_ylabel(r"$q_y$")
    axbc.set_title(
        rf"Berry curvature $\Omega(\mathbf{{q}})$,  $\kappa={kappa}$"
        "\n"
        rf"FHS Chern number  $C = {C:+d}$  "
        r"$\left(C = \frac{1}{2\pi}\iint_{\mathrm{BZ}}\Omega\,\mathrm{d}^2q\right)$"
    )

    # --- right: C vs κ -------------------------------------------------------
    axch.axhline( 0, color="gray",      lw=0.8)
    axch.axvline( 0, color="gray",      lw=0.8)
    axch.axhline( 1, color="steelblue", lw=0.8, ls="--", alpha=0.5)
    axch.axhline(-1, color="steelblue", lw=0.8, ls="--", alpha=0.5)
    axch.scatter(kappa_sweep, C_sweep, color="steelblue", s=70, zorder=5)
    axch.scatter([kappa], [C], color="crimson", s=90, zorder=6,
                 label=rf"$\kappa={kappa}$, $C={C:+d}$")
    axch.set_xlabel(r"$\kappa$")
    axch.set_ylabel(r"Chern number $C$")
    axch.set_yticks([-1, 0, 1])
    axch.set_xlim(-0.25, 0.25)
    axch.set_ylim(-1.6, 1.6)
    axch.legend(fontsize=9)
    axch.set_title(r"$C = \mathrm{sgn}(\kappa) = \pm 1$ for $\kappa \neq 0$")
    axch.grid(alpha=0.3)

    fig.suptitle(
        r"Topological Chern insulator: $|C|=1$ in gapped B phase  "
        rf"($J_x=J_y=J_z=1$,  $\kappa={kappa}$)",
        fontsize=12
    )
    fig.savefig(PLOTS / "HC_5_chern_number.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] Saved  HC_5_chern_number.pdf   (C = {C:+d})")


# ---------------------------------------------------------------------------
# Plot 6 — Gap opening at the Dirac point: κ=0 vs κ≠0 + 1D cross-section
# ---------------------------------------------------------------------------

@plot(6, "Gap opening at Dirac cone — gapped cone + 1D slice")
def plot_gap_opening(Jx=1.0, Jy=1.0, Jz=1.0, kappa_gap=0.20, radius=1.4,
                     n_cone=80, n_line=600):
    """
    Two-panel figure showing the topological gap opening at the Dirac point K.

    Left:   3D cone near K, κ=kappa_gap — gap Δ = 6√3 κ opens; dashed yellow
            line marks the forbidden region [−Δ/2, +Δ/2] at K.
    Right:  1D slice ε(K + δqx ê_x) for κ = 0, 0.05, 0.10, 0.20.
            Solid = ε_+ (upper band), dashed = ε_- = −ε_+ (lower band).
            κ=0 gives a linear (V-shape) cone; κ≠0 rounds the apex into a
            hyperbola with half-gap ε_+(K) = Δ/2 = 3√3 κ (marked by arrows).
    """
    K, _ = KitaevHoneycomb.dirac_points()

    # 2-D mesh centred on K (δq coordinates)
    dq = np.linspace(-radius, radius, n_cone)
    DQX, DQY = np.meshgrid(dq, dq)
    E1 = KitaevHoneycomb(Jx, Jy, Jz, kappa=kappa_gap).dispersion(K[0]+DQX, K[1]+DQY)
    gap_K = 2.0 * float(
        KitaevHoneycomb(Jx, Jy, Jz, kappa=kappa_gap).dispersion(K[0], K[1])
    )   # = 6√3 κ

    # 1-D slice along qx through K
    dqx = np.linspace(-radius, radius, n_line)
    kappa_vals  = [0.00, 0.05, 0.10, 0.20]
    line_colors = ["dimgray", "steelblue", "darkorange", "crimson"]

    fig = plt.figure(figsize=(11, 5.2), layout="constrained")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])

    surf_kw = dict(linewidth=0, antialiased=True, alpha=0.83)

    # --- Left: κ = kappa_gap, gapped ---------------------------------------
    Emax1 = float(np.max(E1))
    ax1.plot_surface(DQX, DQY,  E1, cmap="plasma", **surf_kw)
    ax1.plot_surface(DQX, DQY, -E1, cmap="plasma", **surf_kw)
    # dashed vertical line marking the forbidden gap region at K
    zz = np.linspace(-gap_K/2, gap_K/2, 60)
    ax1.plot([0]*60, [0]*60, zz, color="yellow", lw=2.2, ls="--", zorder=10)
    ax1.scatter(0, 0,  gap_K/2, color="yellow", s=70, depthshade=False, zorder=11)
    ax1.scatter(0, 0, -gap_K/2, color="yellow", s=70, depthshade=False, zorder=11)
    ax1.set_zlim(-Emax1*1.05, Emax1*1.05)
    ax1.set_xlabel(r"$\delta q_x$", labelpad=4)
    ax1.set_ylabel(r"$\delta q_y$", labelpad=4)
    ax1.set_zlabel(r"$\varepsilon$", labelpad=4)
    ax1.set_title(
        rf"$\kappa = {kappa_gap}$" + "\n"
        + rf"$\Delta = 6\sqrt{{3}}\,\kappa \approx {gap_K:.2f}$  (gapped)",
        pad=6,
    )
    ax1.view_init(elev=25, azim=225)

    # --- Right: 1-D cross-section for several κ ----------------------------
    for kv, col in zip(kappa_vals, line_colors):
        m   = KitaevHoneycomb(Jx, Jy, Jz, kappa=kv)
        eps = m.dispersion(K[0] + dqx, K[1])
        lbl = rf"$\kappa = {kv:.2f}$"
        ax2.plot( dqx,  eps, color=col, lw=2.0, label=lbl)
        ax2.plot( dqx, -eps, color=col, lw=2.0, ls="--")
        # annotate the half-gap at δqx = 0 (only for κ > 0)
        if kv > 0:
            half = float(m.dispersion(K[0], K[1]))
            ax2.annotate(
                "", xy=(0, half), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.2),
            )

    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.axvline(0, color="black", lw=0.6, ls=":")
    ax2.set_xlabel(r"$\delta q_x$  along $\mathbf{K}+\delta q_x\hat{e}_x$", fontsize=10)
    ax2.set_ylabel(r"$\varepsilon$", fontsize=10)
    ax2.set_title(
        r"1D cut through $\mathbf{K}$" + "\n"
        + r"solid $= \varepsilon_+$,  dashed $= \varepsilon_- = -\varepsilon_+$",
        pad=6,
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        r"Topological gap opening at $\mathbf{K}$: "
        r"$\varepsilon = \pm\sqrt{|f(\mathbf{q})|^2 + d_z(\mathbf{q})^2}$"
        r"   ($J_x=J_y=J_z=1$)",
        fontsize=11,
    )
    fig.savefig(PLOTS / "HC_6_gap_opening.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] Saved  HC_6_gap_opening.pdf   (Δ at K = {gap_K:.4f}  = 6√3·{kappa_gap})")


# ---------------------------------------------------------------------------
# Plot 9 — Real-space zero-mode: transverse localization of the edge state
# ---------------------------------------------------------------------------

@plot(9, "Zero-mode localization across the ribbon")
def plot_zero_modes(Jx=1.0, Jy=1.0, Jz=1.0, kappa=0.15, M=40):
    """
    The chiral edge modes cross E=0 at k=0: those are the zero modes. At k=0
    the left/right edge states are degenerate, so we disentangle them by
    diagonalizing the row operator m̂ within the two-state E≈0 subspace, giving
    the maximally left- and right-localized zero modes.

    Left:  |ψ(m)|² per row for the two zero modes — one exponentially bound to
           each edge, mirror images. Analogous to the Kitaev-chain MZM plot.
    Right: same profiles on a log axis; the straight-line decay gives the
           localization length ξ set by the bulk gap Δ=6√3κ (ξ ~ v_F/Δ).
    """
    model = KitaevHoneycomb(Jx, Jy, Jz, kappa=kappa)
    val, vec = np.linalg.eigh(model.ribbon_hamiltonian(0.0, M))

    # two states closest to E=0 (the chiral-mode crossing)
    order = np.argsort(np.abs(val))
    idx = order[:2]
    U = vec[:, idx]                                     # (2M, 2)

    # disentangle by the row-position operator within the degenerate subspace
    rows = np.arange(M)
    site_row = np.repeat(rows, 2)                       # row index of each site
    Pmat = (U.conj().T * site_row) @ U                  # 2×2  ⟨ψ_i| m̂ |ψ_j⟩
    _, w = np.linalg.eigh(Pmat)
    psi = U @ w                                         # left- / right-localized

    dens = np.abs(psi)**2
    rowd = dens[0::2, :] + dens[1::2, :]               # (M, 2) per-row density
    rowd = rowd / rowd.sum(0, keepdims=True)
    # order columns so 0 = left-localized, 1 = right-localized
    if (rowd[:, 0] * rows).sum() > (rowd[:, 1] * rows).sum():
        rowd = rowd[:, ::-1]

    sites = np.arange(1, M + 1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    axL.bar(sites - 0.2, rowd[:, 0], width=0.4, color="steelblue", alpha=0.85,
            label="left-edge zero mode")
    axL.bar(sites + 0.2, rowd[:, 1], width=0.4, color="crimson", alpha=0.85,
            label="right-edge zero mode")
    axL.set_xlabel(r"row $m$ across the ribbon width")
    axL.set_ylabel(r"$|\psi(m)|^2$  (normalised)")
    axL.set_title(rf"Zero modes at $k=0$, $E\approx{val[order[0]]:+.1e}$"
                  "\n"
                  r"exponentially bound to opposite edges")
    axL.legend(fontsize=9)
    axL.set_xlim(0, M + 1)

    floor = 1e-6
    axR.semilogy(sites, np.maximum(rowd[:, 0], floor), "o-", ms=3,
                 color="steelblue", label="left-edge zero mode")
    axR.semilogy(sites, np.maximum(rowd[:, 1], floor), "s-", ms=3,
                 color="crimson", label="right-edge zero mode")
    axR.set_xlabel(r"row $m$")
    axR.set_ylabel(r"$|\psi(m)|^2$  (log)")
    axR.set_title(r"Exponential decay: $\xi \sim v_F/\Delta$, "
                  rf"$\Delta=6\sqrt{{3}}\,\kappa={6*np.sqrt(3)*kappa:.2f}$")
    axR.legend(fontsize=9)
    axR.set_xlim(0, M + 1)
    axR.set_ylim(floor, 1)
    axR.grid(alpha=0.3, which="both")

    fig.suptitle(
        rf"Chiral Majorana zero modes of the gapped B phase "
        rf"($\kappa={kappa}$, $M={M}$ rows, $J_x=J_y=J_z=1$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS / "HC_9_zero_modes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [9] Saved  HC_9_zero_modes.pdf   (E_0={val[order[0]]:+.2e}, κ={kappa})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    requested = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(_registry)
    print(f"Running {len(requested)} plot(s): {requested}\n")
    for idx in requested:
        if idx not in _registry:
            print(f"  [!] No plot registered at index {idx}")
            continue
        label, fn = _registry[idx]
        print(f"  [{idx}] {label} ...")
        fn()
    print(f"\nDone — plots in {PLOTS}/")
