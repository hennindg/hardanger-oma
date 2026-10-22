"""
Hardanger Bridge project - coupled aeroelastic model of the antisymmetric modes

Predicts how the frequencies, damping ratios and mode shapes of the first antisymmetric
lateral, vertical and torsional modes change with mean wind speed, when the motion-induced
(self-excited) wind forces of Oiseth et al. (2010) are added, and compares with our data.

Theory (Oiseth, Ronnquist & Sigbjornsson 2010, JWEIA 98, Eqs. 1-5, 14, 16, 18):
  Equation of motion in modal coordinates eta = [lateral y, vertical z, torsion theta]:
      M eta'' + (C0 - Cae(V)) eta' + (K0 - Kae(V)) eta = 0
  M, C0, K0  : still-air modal mass, damping and stiffness (diagonal)
  Cae, Kae   : aerodynamic damping and stiffness from the self-excited forces
      q_y  = 1/2 rho V^2 B   ( p1 y'/V + p2 B th'/V + p3 th + p4 y/B + p5 z'/V + p6 z/B )
      q_z  = 1/2 rho V^2 B   ( h1 z'/V + h2 B th'/V + h3 th + h4 z/B + h5 y'/V + h6 y/B )
      q_th = 1/2 rho V^2 B^2 ( a1 z'/V + a2 B th'/V + a3 th + a4 z/B + a5 y'/V + a6 y/B )
  With frequency-independent ("modified quasi-steady") coefficients, the eigenvalues of the
  state matrix give the frequencies and damping of the coupled system directly at each V.

  Coefficients:
  - h1..h4, a1..a4: modified quasi-steady coefficients fitted to wind tunnel data (Eq. 18)
  - the other terms from quasi-steady theory and the static coefficients (Eq. 16):
      p1 = -2 (D/B) CD,  p5 = CL - (D/B) CD',  p3 = (D/B) CD',  h5 = -2 CL,  a5 = -2 CM,
      p2 = p4 = p6 = h6 = a6 = 0
  - modal coupling between two modes is scaled by the spanwise shape similarity r
    (r = 1 for identical shapes). Oiseth's Eq. 17 gives the product of the two shape ratios,
    which equals r^2 for normalized shapes.

The script first VALIDATES the implementation by recomputing two critical flutter speeds of
Oiseth et al. (2010), Table 6 (design model, modes 4+13 and 6+13), then runs the Hardanger
antisymmetric modes and compares with the identified values.

All literature values: see literature/reference_values.md.

How to use:
  python -m studies.mode_0119.aeroelastic_model
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

AIR_DENSITY = 1.25      # kg/m^3, Oiseth et al. (2010)
B = 18.3                # deck width (m)
D = 3.33                # deck depth (m) used with the coefficients in Oiseth et al. (2010)

# Static force coefficients (Hansen et al. 2006, in Oiseth et al. 2010 below Eq. 16).
# CD is referred to D, CL to B, CM to B^2.
CD_MEAN, CD_SLOPE = 0.70, 0.0
CL_MEAN, CL_SLOPE = -0.25, 2.4
CM_MEAN, CM_SLOPE = 0.01, 0.74

# Modified quasi-steady coefficients (Oiseth et al. 2010, Eq. 18)
H = {1: -2.734, 2: 0.206, 3: 2.271, 4: -0.208}
A = {1: -0.823, 2: -0.258, 3: 0.726, 4: -0.037}

# Hardanger antisymmetric modes: still-air frequency = measured at low wind (Petersen & Oiseth 2017,
# modes 2, 3 and 20), modal equivalent mass / inertia from Oiseth et al. (2010), Table 2
# (modes 2, 3, 18), structural damping 0.5 % (as assumed by Siedziako & Oiseth 2017).
HARDANGER_MODES = [
    {"name": "antisym. lateral", "dof": "y", "freq": 0.105, "mass": 9400.0, "damping": 0.005},
    {"name": "antisym. vertical", "dof": "z", "freq": 0.119, "mass": 12820.0, "damping": 0.005},
    {"name": "antisym. torsion", "dof": "theta", "freq": 0.560, "mass": 426000.0, "damping": 0.005},
]
# Spanwise shape similarity r between modes (1 = identical shapes; all three are one full
# antisymmetric wave over the main span, so 1 is the upper bound of the coupling)
SIMILARITY = 1.0

# Wind speeds for the model curves (m/s)
WIND_SPEEDS = np.linspace(0.0, 30.0, 61)

# Validation against Oiseth et al. (2010), Table 6 (design model, modified quasi-steady coefficients):
# modes 4 (vertical, 0.89 rad/s, 0.6 %) and 13 (torsion, 2.23 rad/s, 0.5 %), similarity product 0.41 -> 93 m/s
# modes 6 (vertical, 1.27 rad/s, 0.7 %) and 13, similarity product 0.57 -> 80 m/s
VALIDATION = [
    {"label": "modes 4 + 13", "expected": 93.0, "omega_expected": 1.40, "similarity_product": 0.41,
     "modes": [{"name": "4", "dof": "z", "freq": 0.89 / (2 * np.pi), "mass": 12820.0, "damping": 0.006},
               {"name": "13", "dof": "theta", "freq": 2.23 / (2 * np.pi), "mass": 426000.0, "damping": 0.005}]},
    {"label": "modes 6 + 13", "expected": 80.0, "omega_expected": 1.66, "similarity_product": 0.57,
     "modes": [{"name": "6", "dof": "z", "freq": 1.27 / (2 * np.pi), "mass": 12820.0, "damping": 0.007},
               {"name": "13", "dof": "theta", "freq": 2.23 / (2 * np.pi), "mass": 426000.0, "damping": 0.005}]},
]

# Our identified values for comparison
MODES_VS_WIND = RESULTS_FOLDER / "step5" / "modes_vs_wind.csv"
TWO_MODE_SSI = RESULTS_FOLDER / "band_ssi_antisym" / "two_modes.csv"
TRACKS = {"antisym. lateral": (0.1051, "lateral"), "antisym. vertical": (0.1171, "vertical")}

OUTPUT_FOLDER = RESULTS_FOLDER / "aeroelastic_model"

# ==================================================================


def self_excited_coefficients():
    """
    All 18 coefficients p1..p6, h1..h6, a1..a6 of the self-excited forces.
    h1-h4 and a1-a4 from Eq. 18; the rest from quasi-steady theory, Eq. 16.
    """
    p = {1: -2 * (D / B) * CD_MEAN,            # lateral force from lateral velocity (drag damping)
         2: 0.0,
         3: (D / B) * CD_SLOPE,                 # lateral force from rotation
         4: 0.0,
         5: CL_MEAN - (D / B) * CD_SLOPE,       # lateral force from vertical velocity
         6: 0.0}
    h = dict(H)
    h[5] = -2 * CL_MEAN                         # vertical force from lateral velocity
    h[6] = 0.0
    a = dict(A)
    a[5] = -2 * CM_MEAN                         # moment from lateral velocity
    a[6] = 0.0
    return p, h, a


def aerodynamic_matrices(V, dofs, r):
    """
    Aerodynamic damping and stiffness matrices (per unit length, modal coordinates) at wind
    speed V for modes whose type is given in dofs (e.g. ["y", "z", "theta"]).
    The force on mode n from the motion of mode m is multiplied by r (shape similarity) for n != m.
    Force = Cae * velocity + Kae * displacement.
    """
    p, h, a = self_excited_coefficients()
    half = 0.5 * AIR_DENSITY
    # Coefficient of force type (row) from velocity / displacement type (column)
    damping = {
        ("y", "y"): half * V * B * p[1], ("y", "z"): half * V * B * p[5], ("y", "theta"): half * V * B * B * p[2],
        ("z", "y"): half * V * B * h[5], ("z", "z"): half * V * B * h[1], ("z", "theta"): half * V * B * B * h[2],
        ("theta", "y"): half * V * B * B * a[5], ("theta", "z"): half * V * B * B * a[1],
        ("theta", "theta"): half * V * B ** 3 * a[2],
    }
    stiffness = {
        ("y", "y"): half * V ** 2 * p[4], ("y", "z"): half * V ** 2 * p[6], ("y", "theta"): half * V ** 2 * B * p[3],
        ("z", "y"): half * V ** 2 * h[6], ("z", "z"): half * V ** 2 * h[4], ("z", "theta"): half * V ** 2 * B * h[3],
        ("theta", "y"): half * V ** 2 * B * a[6], ("theta", "z"): half * V ** 2 * B * a[4],
        ("theta", "theta"): half * V ** 2 * B ** 2 * a[3],
    }
    n = len(dofs)
    Cae = np.zeros((n, n))
    Kae = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            factor = 1.0 if i == j else r
            Cae[i, j] = factor * damping[(dofs[i], dofs[j])]
            Kae[i, j] = factor * stiffness[(dofs[i], dofs[j])]
    return Cae, Kae


def eigen_analysis(V, modes, r):
    """
    Complex eigenvalue problem of the coupled system at wind speed V.
    Returns eigenvalues (one of each conjugate pair) and the displacement parts of the
    eigenvectors (columns).
    """
    m = np.array([mode["mass"] for mode in modes])
    omega0 = 2 * np.pi * np.array([mode["freq"] for mode in modes])
    zeta0 = np.array([mode["damping"] for mode in modes])
    M = np.diag(m)
    C0 = np.diag(2 * zeta0 * omega0 * m)
    K0 = np.diag(omega0 ** 2 * m)
    Cae, Kae = aerodynamic_matrices(V, [mode["dof"] for mode in modes], r)

    n = len(modes)
    Minv = np.linalg.inv(M)
    state = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-Minv @ (K0 - Kae), -Minv @ (C0 - Cae)]])
    lam, vec = np.linalg.eig(state)
    keep = np.imag(lam) > 0
    return lam[keep], vec[:n, keep]


def track_over_wind(modes, r, wind_speeds):
    """
    Frequencies, damping ratios and mode shapes of each mode over the wind speeds, following
    each eigenvalue continuously from still air (closest eigenvalue to the previous one).
    Returns freq, damping [n_speeds, n_modes] and shapes [n_speeds, n_dofs, n_modes] (complex).
    """
    n = len(modes)
    freq = np.zeros((len(wind_speeds), n))
    damping = np.zeros((len(wind_speeds), n))
    shapes = np.zeros((len(wind_speeds), n, n), dtype=complex)
    previous = None
    for k, V in enumerate(wind_speeds):
        lam, vec = eigen_analysis(V, modes, r)
        if previous is None:
            # still air: order by the dominant degree of freedom = the input order
            order = [int(np.argmax(np.abs(vec[i, :]))) for i in range(n)]
        else:
            order = []
            for i in range(n):
                distance = np.abs(lam - previous[i])
                distance[order] = np.inf          # each eigenvalue used once
                order.append(int(np.argmin(distance)))
        lam, vec = lam[order], vec[:, order]
        previous = lam
        freq[k] = np.abs(lam) / (2 * np.pi)
        damping[k] = -np.real(lam) / np.abs(lam)
        shapes[k] = vec
    return freq, damping, shapes


def flutter_speed(modes, r, v_max=200.0, step=0.1):
    """First wind speed where one mode gets zero (then negative) total damping. Returns V and omega."""
    for V in np.arange(step, v_max, step):
        lam, _ = eigen_analysis(V, modes, r)
        zeta = -np.real(lam) / np.abs(lam)
        if np.min(zeta) < 0:
            return V, np.abs(lam[np.argmin(zeta)])
    return np.nan, np.nan


def read_identified():
    """Identified frequency and damping (step 5, full band) and 2-mode SSI results with coupling."""
    full = {name: [] for name in TRACKS}
    with open(MODES_VS_WIND, newline="") as f:
        for row in csv.DictReader(f):
            for name, (track_freq, kind) in TRACKS.items():
                if abs(float(row["track_median_freq_Hz"]) - track_freq) < 0.002 and row["type"] == kind:
                    full[name].append((float(row["mean_wind_speed_ms"]), float(row["freq_Hz"]),
                                       float(row["damping_percent"])))
    two = {name: [] for name in TRACKS}
    with open(TWO_MODE_SSI, newline="") as f:
        for row in csv.DictReader(f):
            name = "antisym. lateral" if row["mode"] == "lateral" else "antisym. vertical"
            two[name].append((float(row["mean_wind_speed_ms"]), float(row["freq_Hz"]),
                              float(row["damping_percent"]), float(row["coupling_amplitude"]),
                              float(row["coupling_phase_deg"])))
    return {k: np.array(v) for k, v in full.items()}, {k: np.array(v) for k, v in two.items()}


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ---- 1. Validation against Oiseth et al. (2010), Table 6 ----
print("Validation: critical flutter speed with modified quasi-steady coefficients (Oiseth 2010, Table 6)")
for case in VALIDATION:
    r = np.sqrt(case["similarity_product"])
    V_cr, omega_cr = flutter_speed(case["modes"], r)
    print(f"  {case['label']}: this model {V_cr:.1f} m/s, omega {omega_cr:.2f} rad/s;   "
          f"Oiseth: {case['expected']:.0f} m/s, {case['omega_expected']:.2f} rad/s")

# ---- 2. Hardanger antisymmetric modes ----
names = [m["name"] for m in HARDANGER_MODES]
f_c, z_c, s_c = track_over_wind(HARDANGER_MODES, SIMILARITY, WIND_SPEEDS)     # coupled
f_u, z_u, s_u = track_over_wind(HARDANGER_MODES, 0.0, WIND_SPEEDS)            # uncoupled (r = 0)
V_cr, omega_cr = flutter_speed(HARDANGER_MODES, SIMILARITY)
print(f"\nHardanger antisymmetric modes (r = {SIMILARITY}): flutter at {V_cr:.0f} m/s "
      f"({omega_cr / (2 * np.pi):.3f} Hz)")

print("\n  U [m/s]   mode                 f coupled [Hz]  f uncoupled   zeta coupled [%]  zeta uncoupled [%]")
for V in [0, 5, 10, 15, 20, 25]:
    k = int(np.argmin(np.abs(WIND_SPEEDS - V)))
    for i, name in enumerate(names):
        print(f"  {V:5.0f}     {name:18s}   {f_c[k, i]:.4f}         {f_u[k, i]:.4f}        "
              f"{100 * z_c[k, i]:6.2f}            {100 * z_u[k, i]:6.2f}")

# Coupling in the mode shapes: vertical motion in the lateral mode, lateral motion in the vertical mode
k20 = int(np.argmin(np.abs(WIND_SPEEDS - 20)))
c_lat = s_c[:, 1, 0] / s_c[:, 0, 0]       # z / y in the lateral mode
c_ver = s_c[:, 0, 1] / s_c[:, 1, 1]       # y / z in the vertical mode
print(f"\nModel mode shapes at 20 m/s: lateral mode |z/y| = {abs(c_lat[k20]):.3f} "
      f"({np.degrees(np.angle(c_lat[k20])):+.0f} deg);  vertical mode |y/z| = {abs(c_ver[k20]):.3f} "
      f"({np.degrees(np.angle(c_ver[k20])):+.0f} deg)")

# ---- 3. Comparison with the identified values ----
full, two = read_identified()
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout="constrained")
for row, (i, name) in enumerate([(0, "antisym. lateral"), (1, "antisym. vertical")]):
    ax_f, ax_z, ax_c, ax_p = axes[row]
    ax_f.plot(full[name][:, 0], full[name][:, 1], "x", color="0.5", label="full-band SSI (step 5)")
    ax_f.plot(two[name][:, 0], two[name][:, 1], "o", color="tab:blue", markersize=4, label="2-mode SSI")
    ax_f.plot(WIND_SPEEDS, f_c[:, i], "k-", label=f"model, coupled (r = {SIMILARITY})")
    ax_f.plot(WIND_SPEEDS, f_u[:, i], "k--", label="model, uncoupled")
    ax_z.plot(full[name][:, 0], full[name][:, 2], "x", color="0.5")
    ax_z.plot(two[name][:, 0], two[name][:, 2], "o", color="tab:blue", markersize=4)
    ax_z.plot(WIND_SPEEDS, 100 * z_c[:, i], "k-")
    ax_z.plot(WIND_SPEEDS, 100 * z_u[:, i], "k--")
    coupling = c_lat if i == 0 else c_ver
    ax_c.plot(two[name][:, 0], two[name][:, 3], "o", color="tab:blue", markersize=4)
    ax_c.plot(WIND_SPEEDS, np.abs(coupling), "k-")
    ax_p.plot(two[name][:, 0], two[name][:, 4], "o", color="tab:blue", markersize=4)
    ax_p.plot(WIND_SPEEDS[1:], np.degrees(np.angle(coupling[1:])), "k-")
    other = "vertical/lateral" if i == 0 else "lateral/vertical"
    ax_f.set_ylabel(f"{name.upper()} MODE\nFrequency [Hz]")
    ax_z.set_ylabel("Damping [%]")
    ax_c.set_ylabel(f"|{other}| in the mode")
    ax_p.set_ylabel(f"phase of {other} [deg]")
    ax_p.set_ylim(-180, 180)
    ax_p.set_yticks(range(-180, 181, 45))
    for ax in axes[row]:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
axes[0][0].legend(fontsize=7)
for ax in axes[1]:
    ax.set_xlabel("Mean wind speed [m/s]")
fig.suptitle("Coupled aeroelastic model (Oiseth et al. 2010 modified quasi-steady coefficients, 3 antisymmetric "
             "modes, structural damping 0.5 %) vs. identified values")
fig.savefig(OUTPUT_FOLDER / "model_vs_data.png", dpi=90)

with open(OUTPUT_FOLDER / "model_curves.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["wind_speed_ms"]
    for name in names:
        header += [f"{name} freq coupled", f"{name} damping coupled %", f"{name} freq uncoupled",
                   f"{name} damping uncoupled %"]
    writer.writerow(header)
    for k, V in enumerate(WIND_SPEEDS):
        line = [f"{V:.1f}"]
        for i in range(len(names)):
            line += [f"{f_c[k, i]:.5f}", f"{100 * z_c[k, i]:.3f}", f"{f_u[k, i]:.5f}", f"{100 * z_u[k, i]:.3f}"]
        writer.writerow(line)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()
