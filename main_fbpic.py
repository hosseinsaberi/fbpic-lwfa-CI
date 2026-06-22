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
#from plasma_dens import gasjet_dens_ramp, capillary_dens_ramp_gaussian, capillary_dens_ramp_parabolic

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
Nz   = 500                                             # Number of grid points along z (longitudinal direction)
zmin = -30.0e-6                                        # Left boundary of the box in z (meters)
zmax = 10.0e-6                                         # Right boundary of the box in z (meters)
dz = (zmax - zmin) / Nz                                # Cell size; dz should resolve laser/plasma 

# Radial box (r)
Nr   = 200                                             # Number of grid points along r (radial direction) - 20–40 cells across rb
rmax = 100e-6 #20.0e-6                                           # Radial box size (meters): 0 <= r <= rmax
dr = rmax / Nr                                         # Cell size

# Azimuthal modes (quasi‑cylindrical)
Nm = 2                                                 # For example Nm = 2 → 2 modes → m = 0 and m = 1

# Time step (based on dz/c for PSATD)
dt = (zmax - zmin) / (Nz * c)                          # dz = (zmax - zmin)/Nz → dt = dz/c


# ============================================================
#                   BACKGROUND PLASMA SETUP
# ============================================================
plasma_type = 'gas_jet_parabolic_radiator' # gas_jet, capillary, capillary-2stage, gas_jet_parabolic, gas_jet_parabolic_radiator
p_zmin = 0.0e-6                                       # Start position of plasma region (meters)
#p_zmax = 500.0e-6                                    # End position of plasma region (meters) ✅ If not defined, Plasma has no hard cutoff
#p_rmax = 18e-6	                                     # Radial extent of plasma (meters) ✅ If not defined, Plasma fills the entire radial domain

# Macro-particle sampling per cell along z, r and theta
p_nz_He = 2     	 # or 8#
p_nz_N = 4     	     # Fot N is higher because beam quality depends on sampling of ionization electrons.
p_nz_ele = p_nz_He     	 # as He is main plasma
p_nr = 2      	 # or 8
p_nt = 4     	 # optional: 8–16 for smoother plasma noise
		 # p_nt ≥ 4 × Nm for proper mode resolution.

# ###################
# Plasma density ramp
# ###################
# Background plasma density (m^-3)
if plasma_type == 'gas_jet':
    n_He = 3.0e18 * 1e6     # Density of Helium atoms
    n_N =  0.01 * n_He      # Density of Nitrogen atoms (1% / 99%)
    
    # Simulation window is initiated outside of the plasma
    ramp_start = 0.e-6
    ramp_length = 20.e-6 # Lramp∼5λp to 20λp; Too short ramp (sharp boundary) Too long ramp laser diffracts

    # Select L_dump such that you have 50 dump data # General rule: 50–200 dumps is ideal
    L_interact = 1.2e-3                                 # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
    L_dump     = 200e-6                                # Desired spatial diagnostic interval along propagation (meters) based on L_interact
elif plasma_type == 'gas_jet_parabolic':
    n_He = 3.0e18 * 1e6     # Density of Helium atoms
    n_N =  0.01 * n_He      # Density of Nitrogen atoms (1% / 99%)
    n_axis = 1/3.  # adjust to control channel depth
    
    # Simulation window is initiated outside of the plasma
    ramp_start = 0.e-6
    ramp_length = 20.e-6 # Lramp∼5λp to 20λp; Too short ramp (sharp boundary) Too long ramp laser diffracts
    
    R_cap = 90e-6       # [m] capillary radius - ✅ realistic 150e-6 to 200e-6
    
    # Select L_dump such that you have 50 dump data # General rule: 50–200 dumps is ideal
    L_interact = 6e-3                                 # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
    L_dump     = 120e-6                                # Desired spatial diagnostic interval along propagation (meters) based on L_interact 
elif plasma_type == 'gas_jet_parabolic_radiator':
    n_He = 3.0e18 * 1e6     # Density of Helium atoms
    n_N =  0.01 * n_He      # Density of Nitrogen atoms (1% / 99%)
    n_axis = 1/3.  # adjust to control channel depth
    
    # Simulation window is initiated outside of the plasma
    ramp_start = 0.e-6
    ramp_length = 20.e-6 # Lramp∼5λp to 20λp; Too short ramp (sharp boundary) Too long ramp laser diffracts
    
    R_cap = 90e-6       # [m] capillary radius - ✅ realistic 150e-6 to 200e-6
    
    # Select L_dump such that you have 50 dump data # General rule: 50–200 dumps is ideal
    L1 = 6e-3 # capillary
    L_interact = 6e-3 + 1e-3                                # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
    L_dump     = 120e-6                                # Desired spatial diagnostic interval along propagation (meters) based on L_interact
    n_max  = 2
    
elif plasma_type == 'capillary' or 'capillary-2stage':
    n_He = 3e18 * 1e6     # Density of Helium atoms
    n_N =  0.01 * n_He      # Density of Nitrogen atoms (1% / 99%)
    n_axis = 1/3.  # adjust to control channel depth
    
    # Simulation window is initiated outside of the plasma
    ramp_start = 0.e-6
    ramp_length = 20.e-6 # Lramp∼5λp to 20λp; Too short ramp (sharp boundary) Too long ramp laser diffracts

    #r_width = 20.e-6     # [m] Gaussian radius (controls transverse density width) - r_width ≈ w0
    
    R_cap = 90e-6       # [m] capillary radius - ✅ realistic 150e-6 to 200e-6
                         # Even though it extends beyond the box, this is physically correct:
                         # Your plasma profile will simply be cut at the boundary Still valid if you're modeling the central channel region

    L_interact = 1e-2                                 # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
    L_dump     = 200e-6                                  # Desired spatial diagnostic interval along propagation (meters) based on L_interact
    if plasma_type == 'capillary-2stage':
        L_flat = L_interact - (ramp_start+ramp_length)
        ramp2_length = ramp_length
        n_max  = 10
        L_interact = L_interact+2e-3 #2.2cm         # Physical interaction length to simulate (meters) - be aware of plasma length as well: p_zmax
        L_dump     = 200e-6                                  # Desired spatial diagnostic interval along propagation (meters) based on L_interact
        
        
        

# ###########################
# Plasma density ramp in HOFI
# ###########################
# Background plasma density (m^-3)
if plasma_type == 'dens_ramp_HeN_hofi':
    n_He = 3.0e18 * 1e6     # Density of Helium atoms
    n_N =  0.01 * n_He      # Density of Nitrogen atoms (1% / 99%)
    
    # Simulation window is initiated outside of the plasma
    ramp_start = 0.e-6
    ramp_length = 50.e-6  # Lramp∼5λp to 20λp; Too short ramp (sharp boundary) Too long ramp laser diffracts
    
    # HOFI channel parameters (physically meaningful)
    n_axis   = 1.0        # on-axis normalized density
    r0      = 10e-6   # MUST be ~ w0 (very important!)
    delta_n = 0.05    # stronger channel (5%)
    r_cap   = 40e-6   # ~3–4 × r0

# ============================================================
#                     PLASMA DENSITY FUNC
# ============================================================
def dens_ramp_HeN_hofi(z, r):
    """Returns normalized plasma density n(z, r) for a HOFI channel"""

    # -----------------------
    # Longitudinal density (your ramp)
    # -----------------------
    nz = np.ones_like(z)

    # linear ramp
    nz = np.where(z < ramp_start + ramp_length,
                  (z - ramp_start) / ramp_length, nz)

    # zero before plasma
    nz = np.where(z < ramp_start, 0.0, nz)

    # -----------------------
    # Radial density (HOFI channel)
    # -----------------------
    # Ideal parabolic channel near axis
    nr = n_axis + delta_n * (r**2 / r0**2)

    # Smooth saturation (physically realistic outer profile)
    # prevents density going to infinity
    nr = n_axis + delta_n * (r**2 / r0**2) * np.exp(-(r**2) / r_cap**2)

    # -----------------------
    # Total density
    # -----------------------
    return nz * nr


def gasjet_dens_ramp(z, r):
    """
    Compute relative gas jet density profile with a linear axial ramp.

    Parameters
    ----------
    z : array_like
        Longitudinal coordinate.
    r : array_like
        Radial coordinate (unused here, included for interface consistency).

    Returns
    -------
    n : ndarray
        Relative density profile n(z), normalized to 1 after the ramp.
    
    Notes
    -----
    - Density increases linearly from 0 to 1 over the ramp length.
    - Density is zero before the start of the ramp.
    - Assumes global parameters:
        ramp_start : float
        ramp_length : float
    """
    # Start with uniform density = 1 everywhere
    n = np.ones_like(z)

    # Apply linear ramp region: 0 → 1 over ramp_length
    n = np.where(z < ramp_start + ramp_length,
                 (z - ramp_start) / ramp_length,
                 n)

    # Enforce zero density before ramp start
    n = np.where(z < ramp_start, 0.0, n)

    return n


def capillary_dens_ramp_gaussian(z, r):
    """
    Compute capillary density profile with axial linear ramp and Gaussian radial profile.

    Parameters
    ----------
    z : array_like
        Longitudinal coordinate.
    r : array_like
        Radial coordinate.

    Returns
    -------
    n : ndarray
        2D relative density profile n(z, r).

    Notes
    -----
    Axial profile:
        - Linear ramp from 0 to 1 over ramp_length
        - Zero before ramp_start

    Radial profile:
        - Gaussian: exp(-r^2 / r_width^2)

    Assumes global parameters:
        ramp_start : float
        ramp_length : float
        r_width : float
    """

    # ----- Axial density profile -----
    n_z = np.ones_like(z)
    n_z = np.where(z < ramp_start + ramp_length,
                   (z - ramp_start) / ramp_length,
                   n_z)
    n_z = np.where(z < ramp_start, 0.0, n_z)

    # ----- Radial Gaussian profile -----
    n_r = np.exp(-(r**2) / (r_width**2))

    # Combine axial and radial dependence
    return n_z * n_r


def capillary_dens_ramp_parabolic(z, r):
    """
    Compute capillary discharge plasma density with:
    - axial linear ramp
    - physically correct radial parabolic channel

    Parameters
    ----------
    z : array_like
        Longitudinal coordinate.
    r : array_like
        Radial coordinate.

    Returns
    -------
    n : ndarray
        2D relative density profile n(z, r).

    Physics
    -------
    In a capillary discharge:
    - Plasma is hottest on axis → expands → lower density
    - Plasma is cooler near wall → higher density
    - Result: density MINIMUM at axis, MAXIMUM at wall

    Radial profile is therefore:
        n(r) = n_axis + (n_wall - n_axis)*(r/R_cap)^2

    The axial profile models a gas/plasma ramp:
        - zero before ramp_start
        - linear increase over ramp_length
        - constant afterward

    Assumes global parameters:
        ramp_start : float
        ramp_length : float
        R_cap : float
    """

    # ----- Axial density profile -----
    # Start with full density everywhere
    n_z = np.ones_like(z)

    # Apply linear ramp region
    # Between ramp_start and ramp_start + ramp_length,
    # density increases from 0 → 1
    n_z = np.where(z < ramp_start + ramp_length,
                   (z - ramp_start) / ramp_length,
                   n_z)

    # Before ramp_start → density = 0
    n_z = np.where(z < ramp_start, 0.0, n_z)

    # ----- Radial parabolic profile (physically correct) -----
    # n_axis: minimum density at center (hot plasma core)
    # Increase toward wall due to cooling → higher density
    
    # Parabolic increase from axis to wall
    n_r = n_axis + (1.0 - n_axis) * (r / R_cap) ** 2

    # Ensure numerical safety (no negative or >1 values)
    n_r = np.clip(n_r, 0.0, 1.0)

    # ----- Combine axial and radial profiles -----
    # Final density is product of longitudinal and transverse structure
    return n_z * n_r

#*************************************
def capillary_dens_ramp_parabolic_radiator(z, r):
    """
    Compute plasma density in a capillary discharge with two regions:

    REGION 1 (z < L1):
        - Axial: linear density ramp (0 → 1) followed by flat region
        - Radial: parabolic channel (physically realistic)
            * minimum density at axis (hot core)
            * maximum density at wall (cooler plasma)

    REGION 2 (z >= L1):
        - Density becomes:
            * uniform in radius (no channel structure)
            * 10× higher than baseline plasma

    Parameters
    ----------
    z : array_like
        Longitudinal coordinate.
    r : array_like
        Radial coordinate.

    Returns
    -------
    n : ndarray
        2D density profile n(z, r).

    Assumes global parameters
    -------------------------
    ramp_start : float
        Start position of density ramp.

    ramp_length : float
        Length over which density increases linearly.

    R_cap : float
        Capillary radius.

    n_axis : float
        Relative density at axis (minimum of parabolic profile).

    L1 : float
        Position where plasma transitions to uniform high-density region.
    """

    # ============================================================
    # 1) AXIAL PROFILE (ramp + flat region)
    # ============================================================
    # Initialize with full density (n = 1) everywhere
    n_z = np.ones_like(z)

    # Apply linear ramp:
    # Between ramp_start and ramp_start + ramp_length,
    # density increases smoothly from 0 to 1
    n_z = np.where(z < ramp_start + ramp_length,
                   (z - ramp_start) / ramp_length,
                   n_z)

    # Before ramp_start, density is zero (no plasma)
    n_z = np.where(z < ramp_start, 0.0, n_z)

    # ============================================================
    # 2) RADIAL PROFILE (parabolic channel)
    # ============================================================
    # Physical model:
    # - Hot core → expansion → low density at axis
    # - Cooler edge → compressed → higher density at wall
    #
    # Parabolic form:
    #   n(r) = n_axis + (1 - n_axis)*(r/R_cap)^2
    n_r = n_axis + (1.0 - n_axis) * (r / R_cap) ** 2

    # Ensure no negative or unphysical values
    n_r = np.clip(n_r, 0.0, 1.0)

    # Combine axial and radial dependence
    n = n_z * n_r

    # ============================================================
    # 3) HIGH-DENSITY UNIFORM REGION (z >= L1)
    # ============================================================
    # After position L1:
    # - Remove radial structure (uniform plasma)
    # - Increase density by factor of 10
    mask_after = z >= L1

    n[mask_after] = n_max

    # ============================================================
    # Final result
    # ============================================================
    return n
#**************************************

def capillary_dens_two_stage(z, r):
    """
    Capillary density with:
      - First linear upramp (0 → 1)
      - Flat region (1)
      - Second ramp (1 → n_max)
      - Final flat region (n_max) up to L_interact
      - Zero beyond interaction length
      - Parabolic radial profile

    Global parameters required:
        ramp_start
        ramp_length
        L_flat
        ramp2_length
        L_interact       # base interaction length
        R_cap

    Internal constants:
        n_max = 10
    """

    # Extend interaction length (your requirement)
    L_total = L_interact 

    # ----- Define transition points -----
    z0 = ramp_start
    z1 = z0 + ramp_length
    z2 = z1 + L_flat
    z3 = z2 + ramp2_length
    z4 = L_total

    # ----- Axial profile -----
    n_z = np.zeros_like(z)

    # Region 1: ramp1 (0 → 1)
    mask1 = (z >= z0) & (z < z1)
    n_z[mask1] = (z[mask1] - z0) / ramp_length

    # Region 2: flat (1)
    mask2 = (z >= z1) & (z < z2)
    n_z[mask2] = 1.0

    # Region 3: ramp2 (1 → n_max)
    mask3 = (z >= z2) & (z < z3)
    n_z[mask3] = 1.0 + (n_max - 1.0) * (z[mask3] - z2) / ramp2_length

    # Region 4: final flat (n_max)
    mask4 = (z >= z3) & (z < z4)
    n_z[mask4] = n_max

    # Region 5: z >= z4 → stays zero

    # ----- Radial profile -----
    n_r = 1.0 - (r / R_cap) ** 2
    n_r = np.clip(n_r, 0.0, 1.0)

    return n_z * n_r
    
# Select plasma density function
if plasma_type == 'gas_jet':
    dens_func = gasjet_dens_ramp
elif plasma_type == 'gas_jet_parabolic':
    dens_func = capillary_dens_ramp_parabolic
elif plasma_type == 'gas_jet_parabolic_radiator':
    dens_func = capillary_dens_ramp_parabolic_radiator
elif plasma_type == 'capillary':
    dens_func = capillary_dens_ramp_parabolic
    #dens_func = capillary_dens_ramp_Gaussian
elif plasma_type == 'capillary_2stage':
    dens_func = capillary_dens_two_stage
# ============================================================
#                     LASER DRIVER
# ============================================================
if plasma_type in ['gas_jet', 'gas_jet_parabolic', 'gas_jet_parabolic_radiator']:
    a0 = 3.          # Laser amplitude
    w0 = 20.e-6 #10.e-6       # Laser waist ~ matched spot size
    tau = 20.e-15     # Laser duration < lambdap
    z0 = -5.e-6      # Laser centroid
    z_foc = 20.e-6   # Focal position
    lambda0 = 0.8e-6 # laser wavelength 
elif plasma_type in ['capillary', 'capillary-2stage']:
    a0 = 2.          # Laser amplitude
    w0 = 20.e-6       # Laser waist ~ matched spot size
    tau = 20.e-15     # Laser duration < lambdap
    z0 = -5.e-6      # Laser centroid
    z_foc = 20.e-6   # Focal position
    lambda0 = 0.8e-6 # laser wavelength 
    
Laser_intensity = (a0 / 0.855 / (lambda0/1e-6) )**2 * 1e18 # [W/cm2]
Laser_intensity_Wm2 = Laser_intensity * 1e4
Laser_power = (pi * w0**2 / 2.) * Laser_intensity_Wm2 # equation for a gaussian beam
Laser_energy = Laser_power * tau
Laser_Rayleigh_length = pi * w0**2 / lambda0  # [um]

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
print('Plasma type is ', plasma_type)
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
    atoms_He = sim.add_new_species(
        q=e,              # charge = +1e → He⁺
        m=4.*m_p,         # mass = 4 proton masses (helium nucleus)
        n=n_He,           # number density
        dens_func=dens_func,
        p_nz=p_nz_He, p_nr=p_nr, p_nt=p_nt,
        p_zmin=p_zmin
    )
    
    atoms_N = sim.add_new_species(
        q=5*e,              # charge = +5e → N⁵⁺
        m=14.*m_p,          # nitrogen ion mass (≈14 atomic mass units)
        n=n_N,              # nitrogen density
        dens_func=dens_func,# spatial density profile
        p_nz=p_nz_N,          # number of macroparticles in z
        p_nr=p_nr,          # number of macroparticles in r
        p_nt=p_nt,          # number of macroparticles in theta
        p_zmin=p_zmin       # minimum z position for initialization
    )

    # Background electrons from pre-ionization
    # Important: the electron density from N5+ is 5x larger than that from He+
    n_e = n_He + 5*n_N
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=dens_func, p_nz=p_nz_ele, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin )

    # Activate ionization of He ions (for levels above 1 - He+→He2+ + e−).
    atoms_He.make_ionizable(
        'He',                # element name (helium)
        target_species=elec, # Store the created electrons in the species `elec`
        level_start=1        # initial state (neutral He)
    )

    # Activate ionization of N ions (for levels above 5).
    # Store the created electrons in a new dedicated electron species that
    # does not contain any macroparticles initially
    elec_from_N = sim.add_new_species( q=-e, m=m_e ) # 👉 Nitrogen is used for ionization injection (the beam you care about)
    atoms_N.make_ionizable( 
        'N', 
        target_species=elec_from_N, 
        level_start=5 
    )

    # --------------------------------------
    # Add laser pulse
    # --------------------------------------
    # Load initial fields
    # Create a Gaussian laser profile
    laser_profile = GaussianLaser(a0, w0, tau, z0, zf=z_foc)
    # Add the laser to the fields of the simulation
    add_laser_pulse( sim, laser_profile)    

    # --------------------------------------
    # Synchrotron
    # --------------------------------------
    if use_synchrotron:
        synchrotron(
            sim=sim,
            species=elec_from_N,
            dic_species={"trapped_elec": elec_from_N},
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

    d = particle_diags(sim, diag_period, species={"trapped_elec": elec_from_N, "plasma_elec": elec})
    sim.diags.append(d)

    d = particle_charge_density_diags(sim, diag_period, species_dict={"trapped_elec": elec_from_N, "plasma_elec": elec})
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
