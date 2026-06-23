# beam_fbpic.py

from scipy.constants import c, e, m_e, epsilon_0
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

# ============================================================
#                      ELECTRON DRIVER PARAMETERS
# ============================================================
q = -e                                               # Particle charge (C): electrons are negative
m = m_e                                              # Particle mass (kg): electron rest mass

# Beam size
sig_r  = 50e-6                                       # Transverse RMS size (meters)
sig_z  = 15e-6                                       # Longitudinal RMS size (meters)
n_emit = 2e-6                                        # normalized emittance (m·rad)

# Beam energy
Eb = 250                                             # Beam energy [MeV]
gamma0    = Eb / 0.511                               # Mean Lorentz factor (energy ≈ 0.511 MeV * gamma)
dE_E = 0.01
sig_gamma = dE_E * gamma0                             # RMS energy spread expressed directly in Lorentz-factor units (γ)
                                                          # Most particles will fall between 480 and 500 (±10).
                                                          # Some go further out because it’s a Gaussian.
                                                     # The actual energy spread is ΔE = sig_gamma * 0.511 MeV (≈ 5.1 MeV here).
                                                     # Relative energy spread is ΔE/E0 ≈ Δγ/γ0 usually expressed in [%]

# Macro/real particle counts
# FBPIC computes the weight of each macroparticle as w=n_real/n_macro (which is electrons per macroparticle)
# ★ Increase beam macroparticles (reduces statistical noise)
# Noise ∝ 1 / √(N_macro) ... ensure to be arounce ~0.2% or less
n_real  = 9.4e8                                        # Number of physical electrons represented (for charge weighting)
#n_macro = int(2e5)     # 200k   (good)               # Number of macroparticles to sample the beam distribution
n_macro = int(1e6)     # 500k   (very smooth spectra)

# Beam focus (Where and when the beam reaches its smallest transverse radius.)
zf = -60e-6                                          # Target longitudinal focus position of the beam (meters)
tf = 0.0                                             # Time at which the beam is focused at zf (seconds)

# =========================================================
# Add driver beam (Gaussian-distributed relativistic bunch)
# =========================================================

def add_driver(sim):
	driver = add_particle_bunch_gaussian(
	    sim=sim,                             # (Simulation) Target FBPIC Simulation object

	    # --- Required species properties ---
	    q=q,                                 # (float, Coulomb) Particle charge; for electrons: q = -e
	    m=m,                                 # (float, kg)      Particle mass; for electrons: m = m_e

	    # --- Beam sizes (RMS) ---
	    sig_r=sig_r,                         # (float, m)  Transverse RMS size σ_r
	    sig_z=sig_z,                         # (float, m)  Longitudinal RMS size σ_z

	    # --- Emittance and energy ---
	    n_emit=n_emit,                       # (float, m)  Normalized emittance ε_n
	    gamma0=gamma0,                       # (float)     Mean Lorentz factor ⟨γ⟩
	    sig_gamma=sig_gamma,                 # (float)     Absolute RMS energy spread in γ (NOT relative)

	    # --- Macro/real particle accounting ---
	    n_physical_particles=n_real,         # (float)     Number of physical particles represented by the bunch
	    n_macroparticles=n_macro,            # (int)       Number of macroparticles sampled for PIC

	    # --- Focusing settings (ballistic, space-charge not accounted in optics) ---
	    tf=tf,                               # (float, s)  Time at which the bunch comes to focus (t = 0 is load time)
	    zf=zf,                               # (float, m)  Longitudinal position of the focus (lab frame)

	    # --- (Optional) Boosted-frame configuration ---
	#    boost=boost,                         # (BoostConverter or None) Lorentz boost of the simulation; None for lab frame

		    # --- (Optional) Save initial distribution ---
	#    save_beam=save_beam,                 # (str or None) If set, saves initial beam phase space to '<save_beam>.npz'

		    # --- (Optional) Ballistic injection plane (useful in boosted frame) ---
	#    z_injection_plane=z_injection_plane, # (float or None, m; lab frame) For z < this, motion is ballistic

		    # --- (Optional) Self fields and symmetry ---
	    initialize_self_field=True,          # (bool)  If True, compute and deposit bunch space-charge fields on grid
	    symmetrize=True                      # (bool)  4-fold rotational symmetry to cancel initial (x,y) offsets
    						 # Use symmetrize=True for clean, noise‑free, on‑axis beams
    						 # Set symmetrize=False for realistic beams or instability studies
	)
	return driver
