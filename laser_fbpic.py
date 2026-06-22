# laser_fbpic.py

import numpy as np
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser import GaussianLaser

# ================================================================
#                        LASER PARAMETERS
# ================================================================
a0  = 4.0                                # Normalized vector potential (dimensionless laser amplitude)
w0  = 5e-6                               # Laser waist (radius at 1/e^2 intensity), in meters
tau = 16e-15                             # Laser duration (FWHM in time for envelope models), in seconds
z0  = -20e-6                             # Laser focus position along z (meters) - start before plasma
zf  = 0.0        			 # [m] Focal plane position (lab frame) - focus at plasma entrance
lambda0=0.8e-6         			 # [m] Central wavelength (default 0.8 μm for Ti:Sapphire).
theta_pol=0.0	       			 # [rad] Linear polarization angle relative to x-axis (0) and y-axis (pi/2). 



# ================================================================
#                        LASER PROFILE
# ================================================================
# Example laser profiles: keep commented out unless you want to inject a laser field

# -------------------------------------------
# Gaussian laser profile: Linear polarization
# -------------------------------------------
# Define a linearly-polarized Gaussian laser profile.
# All quantities are in the LAB frame (unless you pass gamma_boost in add_laser_pulse).
laser_profile_linear_pol = GaussianLaser(          
	a0=a0,			# [dimensionless] Peak normalized vector potential at the focal plane.
                      		# a0 ≈ 0.85 * sqrt(I[10^18 W/cm^2]) * λ0[μm]
                      		# Example a0=2 corresponds to ~5.5e18 W/cm² @ 0.8 μm.
	
	waist=w0,               # [m] Waist at the focal plane (1/e^2 radius of intensity).
                      		# Choose large enough to be resolved by Nr, Δr.

	tau=tau,                # [s] Pulse duration (intensity FWHM for a Gaussian envelope convention in FBPIC).
                     		# Must be compatible with Δt (time step) and Nz, Δz.
        
	z0=z0                   # [m] Initial centroid position of the pulse envelope in the box (lab frame).
                      		# If zf is not given, zf defaults to z0 (i.e., pulse starts at focus).
        
        # --- OPTIONAL PARAMETERS (with defaults shown) ---
	zf=None		       # [m] Focal plane position (lab frame). If None, zf = z0 (focused at initialization).
	theta_pol=0.0	       # [rad] Linear polarization angle relative to x-axis. 0 → Ex-polarized.
	lambda0=lambda0        # [m] Central wavelength (default 0.8 μm for Ti:Sapphire).
	cep_phase=0.0 	       # [rad] Carrier-Envelope Phase (phase of carrier where envelope is max).
	phi2_chirp=0.0         # [s^2] Group delay dispersion at focus. >0 = positive chirp (red leading, blue trailing).
	prop_dir    = 1        # [+1 or -1] Propagation direction: +1 → +z, -1 → -z.
)


# ---------------------------------------------
# Gaussian laser profile: Circular polarization
# ---------------------------------------------
a0_total = a0
a0_each  = a0_total / np.sqrt(2)   # each component has a0 / √2

# X-POLARIZED component (cos phase)
# theta_pol = 0   → linear polarization along x
# cep_phase = 0   → cos(ωt)
# --------------------------------------------------------
laser_x = GaussianLaser(
    a0=a0_each,
    waist=w0,
    tau=tau,
    z0=z0,
    zf=zf,
    theta_pol=0.0,          # x-polarized
    lambda0=lambda0,
    cep_phase=0.0,          # cos(ωt)
    phi2_chirp=0.0,
    propagation_direction=1
)

# Y-POLARIZED component (sin phase)
# theta_pol = π/2 → linear polarization along y
# cep_phase = +π/2 → sin(ωt) = cos(ωt + π/2)
# --------------------------------------------------------
laser_y = GaussianLaser(
    a0=a0_each,
    waist=w0,
    tau=tau,
    z0=z0,
    zf=None,
    theta_pol=np.pi/2,      # y-polarized
    lambda0=lambda0,
    cep_phase=np.pi/2,      # 90° phase shift → circular polarization
    phi2_chirp=0.0,
    propagation_direction=1
)

# Add both components to the simulation
# --------------------------------------------------------
laser_profile_circular_pol = laser_x + laser_y




# ================================================================
#                           ADD LASER
# ================================================================
laser_profile = laser_profile_linear_pol
add_laser_pulse(
    sim,			# The Simulation object (must already exist)
    laser_profile,		# Any valid FBPIC laser profile object
#    gamma_boost=20   		# (Optional) Lorentz factor if using boosted frame
#    method='antenna',		# 'direct'  (default) places laser on grid instantly
#                    		#' 'antenna' generates laser progressively from boundary
#    z0_antenna=0.0,      # (Required for antenna) Antenna position (meters, lab frame)
#    v_antenna=0.0        # (Optional) Antenna velocity (m/s, lab frame)
)
