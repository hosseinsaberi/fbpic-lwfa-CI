"""
===============================================================
FBPIC Simulation Script
===============================================================
Purpose:
    LWFA simulation for hard X-ray generation
        - Ramped uniform plasma => NEXT: HOFI Channel
        - Synchrotron radiation & diagnostics enabled
        - Lab frame => NEXT: Boosted beam frame
        - Moving window
        - Quasi‑cylindrical (m = 0,1,...)

Dates: 
    1st version: 05/05/2026
===============================================================
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, pi, m_p

# Import the relevant structures in FBPIC
# #######################################
# FBPIC core
from fbpic.main import Simulation 

# Add a laser pulse to simulation
from fbpic.lpa_utils.laser import add_laser_pulse

# Define Gaussian laser profile
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser


# Smoothing filter
# Class that applies a binomial smoothing filter to electromagnetic fields on the grid
from fbpic.fields.smoothing import BinomialSmoother

# Import the utility that fixes all random number generators (Python, NumPy, FBPIC).
# Setting a random seed ensures reproducible simulations, so Monte‑Carlo processes
# (e.g., ionization, particle loading, Gaussian beam sampling) give identical results
# each time the script is run.
from fbpic.utils.random_seed import set_random_seed
set_random_seed(42)     # The value 42 is arbitrary (a traditional default); any integer would work.

# Import MY custom diagnostic modules
# ###################################
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, ParticleChargeDensityDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint
from diagnostics_fbpic import field_diags, particle_diags, particle_charge_density_diags
from synchrotron_fbpic import synchrotron
from beam_fbpic import add_witness

# ============================================================
#                 SIMULATION GLOBAL SETTINGS
# ============================================================
use_cuda = True        # Enable GPU acceleration (recommended)
# Order of the stencil for z derivatives in the Maxwell solver.
# -1 for infinite order: exact dispersion in all directions (advised for single-GPU/single-CPU simulation).
# Positive even number for finite-order stencil (required for multi-GPU/CPU with MPI).
# Larger n_order → more MPI overhead but more accurate dispersion. Typical trade-off: n_order = 32.
n_order = -1           # Maxwell solver stencil order (-1 = infinite, most accurate)
use_synchrotron = True # Enable synchrotron radiation module


# ============================================================
#                 SIMULATION BOX GEOMETRY
# ============================================================
# Longitudinal box (z in metres)
Nz   = 2000                                             # Number of grid points along z (longitudinal direction)
zmin = -140.0e-6                                        # Left boundary of the box in z (meters)
zmax = 20.0e-6                                         # Right boundary of the box in z (meters)
dz = (zmax - zmin) / Nz                                # Cell size; dz should resolve laser/plasma 

# Radial box (r)
Nr   = 200                                             # Number of grid points along r (radial direction) - 20–40 cells across rb
rmax = 300e-6 #20.0e-6                                           # Radial box size (meters): 0 <= r <= rmax
dr = rmax / Nr                                         # Cell size

# Azimuthal modes (quasi‑cylindrical)
Nm = 2                                                 # For example Nm = 2 → 2 modes → m = 0 and m = 1

# Time step (based on dz/c for PSATD)
dt = (zmax - zmin) / (Nz * c)                          # dz = (zmax - zmin)/Nz → dt = dz/c


# ============================================================
#                   BACKGROUND PLASMA SETUP
# ============================================================
p_zmin = 0.0e-6                                       # Start position of plasma region (meters)
#p_zmax = 500.0e-6                                    # End position of plasma region (meters) ✅ If not defined, Plasma has no hard cutoff
#p_rmax = 18e-6	                                     # Radial extent of plasma (meters) ✅ If not defined, Plasma fills the entire radial domain

# Macro-particle sampling per cell along z, r and theta
p_nz_H = 2     	 # or 8#
p_nz_ele = p_nz_H     	 # as He is main plasma
p_nr = 2      	 # or 8
p_nt = 4     	 # optional: 8–16 for smoother plasma noise
		 # p_nt ≥ 4 × Nm for proper mode resolution.

# ###################
# Plasma density ramp
# ###################
# Background plasma density (m^-3)
# Peak density (convert cm^-3 → m^-3)
n0 = 1e17 * 1e6   # m^-3
    
# Simulation window is initiated outside of the plasma
# Positions (in meters)
z_start  = 0.0
z_ramp_up_end   = 1.0e-3
z_plateau_end   = 6.0e-3
z_ramp_down_end = 7.0e-3

# Select L_dump such that you have 50 dump data # General rule: 50–200 dumps is ideal
L_interact = z_ramp_down_end                                 # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
L_dump     = 0.1e-3                                # Desired spatial diagnostic interval along propagation (meters) based on L_interact

# ============================================================
#                     PLASMA DENSITY FUNC
# ============================================================
def dens_func(z, r):
    """
    Plasma density profile (m^-3) for:
    - 1 mm up-ramp
    - 5 mm plateau
    - 1 mm down-ramp
    Smooth sin^2 ramps
    """

    # Initialize density
    n = np.zeros_like(z)

    # ----------------------
    # Up ramp (0 → 1 mm)
    # ----------------------
    mask_up = (z >= z_start) & (z < z_ramp_up_end)
    xi = (z[mask_up] - z_start) / (z_ramp_up_end - z_start)
    n[mask_up] = n0 * np.sin(0.5 * np.pi * xi)**2

    # ----------------------
    # Plateau (1 → 6 mm)
    # ----------------------
    mask_plateau = (z >= z_ramp_up_end) & (z < z_plateau_end)
    n[mask_plateau] = n0

    # ----------------------
    # Down ramp (6 → 7 mm)
    # ----------------------
    mask_down = (z >= z_plateau_end) & (z < z_ramp_down_end)
    xi = (z[mask_down] - z_plateau_end) / (z_ramp_down_end - z_plateau_end)
    n[mask_down] = n0 * np.cos(0.5 * np.pi * xi)**2

    return n


# ============================================================
#                     LASER DRIVER
# ============================================================
lambda0 = 0.8e-6   # m
tau     = 23.e-15  # s
w0      = 90.e-6   # m
El      = 2.17     # J

P0 = 0.94 * El / tau                         # W
I0_Wm2 = 2 * P0 / (pi * w0**2)              # W/m^2
I0_Wcm2 = I0_Wm2 * 1e-4                     # W/cm^2

a0 = 0.855 * (lambda0 * 1e6) * np.sqrt(I0_Wcm2 / 1e18)

z0 = -20.e-6
z_foc = 0.e-6

Laser_intensity = I0_Wcm2                   # W/cm^2
Laser_power = P0                            # W
Laser_energy = El                           # J
Laser_Rayleigh_length = pi * w0**2 / lambda0  # m

# ============================================================
#                     MOVING WINDOW SETTINGS
# ============================================================
v_window = c                                        # window moving at speed of light


# ============================================================
#                       SIMULATION RUNTIME
# ============================================================
T_interact = (L_interact + (zmax - zmin)) / v_window # Compute time for moving window to traverse plasma + its own length
                                                     # (assumes window starts before entering the plasma region)

# ============================================================
#                          DIAGNOSTICS
# ============================================================
diag_period = int(L_dump / dz)             # Write diagnostics every N time steps
                                           # General rule: 50–200 dumps is ideal
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 100  # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

# ============================================================
#                            SMOOTHER
# ============================================================
# Apply binomial smoothing to fields (more passes in r than z) to reduce numerical noise,
# with compensators to preserve low-frequency physical content
smoother = BinomialSmoother(
    n_passes={'z': 3, 'r': 6},
    compensator={'z': True, 'r': True}
)


# ============================================================
#                            USER DATA TO PRINT
# ============================================================
# Calculate plasma frequency and plasma wavelength
#wp_He = np.sqrt(n_He * e**2 / epsilon_0 / m_e)
#lambda_p_He = 2 * pi * c / wp_He
#skin_depth_He = c / wp_He

#wp_N = np.sqrt(n_N * e**2 / epsilon_0 / m_e)
#lambda_p_N = 2 * pi * c / wp_N
#skin_depth_N = c / wp_N

print("//////////////////////////////////////////////////////////")
print(f"dz = {dz:.2e} m")
print(f"dr = {dr:.2e} m")
#print(f"lambda_p (He) = {lambda_p_He:.2e} m")
#print(f"lambda_p (N) = {lambda_p_N:.2e} m")
#print(f"skin depth (He) = {skin_depth_He:.2e} m")
#print(f"cell per wavelength (lamdap/dz) = {int(lambda_p / dz)}")
#print("====>>> for PWFA you need ≥ 30–60 cells per lambda (excellent)")
#print(f"skin_depdth/dr) = {int(skin_depth / dr)}")
#print("====>>> 20–40 cells per skin depth is typical\n")
print("diag_period =", diag_period)
print("diag_length =", L_dump, 'm')
print("\n\nLASER PARAMETERS")
print(f"Laser intensity = {Laser_intensity:.2e} W/cm2")
print(f"Laser power     = {Laser_power/1e12:.2f} TW")
print(f"Laser energy    = {Laser_energy:.2e} J")
print(f"Laser Rayleigh length = {Laser_Rayleigh_length/1e-6:.2f} um")
print(f"a0     = {a0:.2f}")
print("//////////////////////////////////////////////////////////")

# ################################################################
# ================================================================
#                          MAIN PROGRAM
# ================================================================
# ################################################################
if __name__ == "__main__":               # Standard Python entry point guard

    # --------------------------------------
    # Initialize Simulation Object
    # --------------------------------------
    sim = Simulation(    	
    	# Spatial domain configuration
    	# --------------------------------------------------------
    	Nz, zmax,                     # (int, float) Number of grid cells in z; box size in z (edge of last cell)
    	Nr, rmax,                     # (int, float) Number of grid cells in r; box size in r (edge of last cell)
    	Nm,                           # (int) Number of azimuthal modes (m=0...Nm-1)
    	dt,                           # (float) Time step of the PIC solver
    	zmin=zmin,                    # (float) Position of lower z‑edge of the simulation box (edge of first cell)
		
    	# Maxwell solver configuration
    	# --------------------------------------------------------
    	n_order=n_order,              # (int) Order of finite‑order PSATD stencil; -1 = infinite order
                                      # High n_order → better dispersion but more MPI communication
       
        # Plasma shortcut initialization (disabled unless n_e provided)
        # If disbaled means no pre plasma but gas ionization
    	# --------------------------------------------------------
#    	p_zmin=p_zmin,               # (float) Minimal z for plasma loading if using built‑in plasma creation
#    	p_zmax=p_zmax,               # (float) Max z for plasma loading
#    	p_rmin=0.0,                  # (float) Minimal r for plasma loading
#   	p_rmax=p_rmax,         	     # (float) Max r for plasma loading
#    	p_nz=p_nz,                   # (int) Number of macro‑particles per cell along z (only if n_e used)
#    	p_nr=p_nr,                   # (int) Number of macro‑particles per cell along r
#    	p_nt=p_nt,                   # (int) Number of macro‑particles per θ
#    	n_e=n_e,                     # (float or None) Plasma density (if provided, auto‑creates electrons, if not NO plasma is automatically created) 
#   	dens_func=dens_func,         # (callable or None) Density function for plasma loading

    	# Current / smoothing / numerical stability
    	# --------------------------------------------------------
    	filter_currents=True,        # (bool) Filter charge & current in k‑space (recommended)
    	smoother=smoother,           # (BinomialSmoother or None) Charge/current spatial smoother

    	# Galilean / comoving PSATD solver
    	# --------------------------------------------------------
    	v_comoving=0.99999 * c,      # (float or None) Speed of comoving frame; None = standard PSATD
    	use_galilean=True,           # (bool) Whether to use Galilean PSATD scheme when v_comoving is set

    	# Particles: initialization & deposition
    	# --------------------------------------------------------
#    	initialize_ions=False,       # (bool) Auto‑create ions (H+) along with electrons if n_e set
    	particle_shape='cubic',      # (str) Particle shape: 'linear' (=1st order) or 'cubic' (=3rd order)
    	use_ruyten_shapes=True,      # (bool) Use Ruyten shapes for better near-axis charge deposition
    	use_modified_volume=True,    # (bool) Modified cell volume near axis for correct spectral solver behavior

    	# Parallelization / GPU / boundary exchanges
    	# --------------------------------------------------------
    	use_cuda=use_cuda,           # (bool) Use GPU acceleration
    	use_all_mpi_ranks=True,      # (bool) All MPI ranks participate in one simulation (default)
    	exchange_period=None,        # (int or None) Steps before exchanging particles between MPI domains

    	# Boundary conditions + damping
    	# --------------------------------------------------------
    	boundaries={'z': 'open', 'r': 'reflective'},
                                     # 'z': 'periodic' or 'open' (open = absorbing PML)
                                     # 'r': 'reflective' or 'open' (open = radial PML, more expensive)
    	n_guard=None,                # (int or None) Number of guard cells (auto = 2*n_order, or 64 if infinite order)
    	n_damp={'z': 64, 'r': 32},   # (dict) Number of damping cells for absorbing layers in z and r

    	# Boosted‑frame support
    	# --------------------------------------------------------
    	gamma_boost=None,            # (float or None) Lorentz factor of boosted simulation; all spatial inputs remain LAB values

    	# Diagnostics / verbosity
    	# --------------------------------------------------------
    	verbose_level=1,             # (int) 0=no output; 1=basic; 2=full detailed setup printout
    )


    # --------------------------------------
    # Insert Background Plasma Electrons
    # --------------------------------------
    # (1) Adding pre-ionized ion species
    # Add the Helium ions (pre-ionized up to level 1),
    # the Nitrogen ions (pre-ionized up to level 5)
    # and the associated electrons (from the pre-ionized levels)

    # Helium is already ionized once (He → He⁺)
    # Each helium ion contributes 1 free electron (already accounted for later)
    atoms_H = sim.add_new_species(
        q = e,                # charge = +1e → H⁺ (proton)
        m = 1.0 * m_p,        # hydrogen ion mass = proton mass
        n = n0,               # number density
        dens_func = dens_func,
        p_nz = p_nz_H,       # you can reuse same macro-particle numbers
        p_nr = p_nr,
        p_nt = p_nt,
        p_zmin = p_zmin
    )
    
    # Background electrons from pre-ionization
    # Important: the electron density from N5+ is 5x larger than that from He+
    n_e = n0
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=dens_func, p_nz=p_nz_ele, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin )

    # --------------------------------------
    # Add laser pulse
    # --------------------------------------
    # Load initial fields
    # Create a Gaussian laser profile
    laser_profile = GaussianLaser(a0, w0, tau, z0, zf=z_foc)
    # Add the laser to the fields of the simulation
    add_laser_pulse( sim, laser_profile)    

    # --------------------------------------
    # Add driver beam
    # --------------------------------------
    witness=add_witness(sim)
    
    # --------------------------------------
    # Synchrotron
    # --------------------------------------
    if use_synchrotron:
        synchrotron(
            sim=sim,
            species=witness,
            dic_species={"witness": witness},
            diag_period=diag_period
        )

    # --------------------------------------
    # Handle simulation start mode:
    # --------------------------------------
    # - If starting fresh → optionally enable electron tracking
    # - If restarting → reload full simulation state from checkpoint
    if use_restart is False:
        # Fresh run: enable tracking of electron trajectories if requested
        if track_electrons:
            elec.track(sim.comm)
            elec_from_N.track(sim.comm)
    else:
        # Restart run: load fields and particles from the latest saved checkpoint
        restart_from_checkpoint(sim)

    
    # --------------------------------------
    # Diagnostics
    # --------------------------------------
#    sim.diags = [
#                FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
#                ParticleDiagnostic( diag_period,
#                    {"electrons from N": elec_from_N, "electrons": elec},
#                    comm=sim.comm ),
                # Since rho from `FieldDiagnostic` is 0 almost everywhere
                # (neutral plasma), it is useful to see the charge density
                # of individual particles
#                ParticleChargeDensityDiagnostic( diag_period, sim,
#                    {"electrons": elec} )
#                ]

    d = field_diags(sim, diag_period)
    sim.diags.append(d)

    d = particle_diags(sim, diag_period, species={"witness": witness})
    sim.diags.append(d)

    d = particle_charge_density_diags(sim, diag_period, species_dict={"witness": witness})
    sim.diags.append(d)

    # --------------------------------------
    # Moving window + runtime
    # --------------------------------------
    sim.set_moving_window(v=v_window)    # Enable moving window that follows the laser/beam at speed v

    N_step = int(T_interact / sim.dt)    # Number of time steps to cover the requested interaction time
    print(f"Running simulation for {N_step} iterations…")  # Informative console message

    # --------------------------------------
    # Add periodic checkpoints
    # --------------------------------------
    # Add periodic checkpoints to save simulation state (fields + particles) for restart or recovery
    if save_checkpoints:
        set_periodic_checkpoint(sim, checkpoint_period)
    
    # --------------------------------------
    # Run PIC Loop
    # --------------------------------------
    sim.step(N_step)                     # Advance fields and particles for N_step iterations

    print("Simulation complete.\n")      # Final message on successful completion
