#synchrotron_fbpic.py

from fbpic.openpmd_diag import SynchrotronRadiationDiagnostic
import numpy as np

# ============================================================
#                    SYNCHROTRON RADIATION
# ============================================================
# Synchrotron radiation histogram axes (for emitted photons)
photon_energy_axis = (                     # 3-tuple: (E_min, E_max, N_bins), energies in Joules
    0.1e3 * 1.6e-19,                        # Minimum photon energy (converted from eV → J)
    100e3  * 1.6e-19,                        # Maximum photon energy (converted from eV → J)
    500                                   # Number of energy bins
)
theta_axis = (                           # Angular axis (min, max, bins) in radians
    -0.051,                                 # Minimum angle (rad)
    0.051,                                  # Maximum angle (rad)
    102                                    # Number of angular bins
)


# ----------------------------
# Enable synchrotron‑radiation 
# ----------------------------
def synchrotron(sim, species, dic_species, diag_period):
    """
    Activate synchrotron radiation for the given particle species.
    
    Configures:
    - photon energy histogram (E_min → E_max, N_bins)
    - angular emission histograms in θ_x and θ_y
    - gamma cutoff (minimum γ for emission)
    - optional radiation reaction
    - spectral kernel resolution (x_max, nSamples)
    """

    species.activate_synchrotron(

        # ------------------------------------------------------------------
        # Photon energy axis
        # ------------------------------------------------------------------
        # Format: (E_min, E_max, N_bins), units = Joules.
        #
        # Energy range chosen: 1e-4 eV → 10 eV
        # Converted with 1 eV = 1.6e-19 J.
        #
        # 1000 bins gives reasonably fine resolution for broadband spectra.
        #
        photon_energy_axis = photon_energy_axis,

        # ------------------------------------------------------------------
        # Angular axis in x (horizontal)
        # ------------------------------------------------------------------
        # Format: (θ_min, θ_max, N_bins), units = radians.
        #
        # Range: ±0.05 rad ≈ ±50 mrad.
        # Synchrotron emission is strongly forward-beamed, so small angles
        # cover the emission cone for mildly/ultra-relativistic electrons.
        #
        theta_x_axis = theta_axis,

        # ------------------------------------------------------------------
        # Angular axis in y (vertical)
        # ------------------------------------------------------------------
        # Identical resolution to θ_x for symmetry.
        #
        theta_y_axis = theta_axis,

        # ------------------------------------------------------------------
        # Gamma cutoff
        # ------------------------------------------------------------------
        # Radiation is only computed for electrons with γ above this limit.
        #
        # Reason:
        # Low-energy electrons emit extremely weak synchrotron radiation.
        # Skipping them improves performance with no meaningful loss.
        #
        gamma_cutoff = 2.0,

        # ------------------------------------------------------------------
        # Radiation Reaction
        # ------------------------------------------------------------------
        # If True:
        #   - photon emission reduces the electron's momentum & energy.
        # If False:
        #   - emission is diagnostic only; particle motion ignores losses.
        #
        radiation_reaction = False,

        # ------------------------------------------------------------------
        # x_max: extent of synchrotron kernel sampling
        # ------------------------------------------------------------------
        # Synchrotron spectral functions involve integrals over modified
        # Bessel functions. These integrals decay exponentially, but not
        # instantly, so an upper limit (x_max) must be chosen.
        #
        # Larger x_max → more accurate high-energy tail.
        # Typical values: 10–30. Default here: 20.
        #
        x_max = 30,

        # ------------------------------------------------------------------
        # nSamples: number of samples for the spectral profile
        # ------------------------------------------------------------------
        # The synchrotron kernel F(x) is computed numerically using a
        # tabulated sampling. This value sets the number of sampling points.
        #
        # Higher nSamples → smoother, more accurate spectrum.
        # Lower nSamples → faster but less accurate.
        #
        # Common choices:
        #   512  = fast
        #   1024 = good balance
        #   2048 = high accuracy (default chosen here)
        #
        nSamples = 2048,

        # ------------------------------------------------------------------
        # Lorentz boost (optional)
        # ------------------------------------------------------------------
        # boost = (beta_x, beta_y, beta_z)
        #
        # If provided:
        #   - emitted radiation is Lorentz-transformed to the observer frame.
        # If None:
        #   - radiation deposited in the simulation frame.
        #
        boost = None
    )

    # --------------------------------------------------------------------------
    # Synchrotron Radiation Diagnostic
    # --------------------------------------------------------------------------
    # This diagnostic writes the emitted synchrotron photon data to disk using
    # the openPMD standard. It requires that the species (e.g. `elec`) has already
    # had synchrotron emission activated with elec.activate_synchrotron().
    #
    # What this diagnostic outputs (per iteration):
    #   • Photon energy histogram (using photon_energy_axis defined earlier)
    #   • Angular distribution θx, θy (theta_x_axis, theta_y_axis)
    #   • Total radiated power
    #   • 3D spectral–angular photon distribution
    #
    # The output is written in openPMD format inside write_dir.
    #
    # IMPORTANT:
    #   • All species included here must share identical synchrotron grids.
    #   • If running in parallel, `comm=sim.comm` is REQUIRED so that data is
    #     gathered on rank 0 and guard cells are removed (clean output).
    #   • If comm=None, *each rank writes its own file* and you MUST use
    #     different write_dir names for each rank — usually not recommended.
    #
    # PERIOD OPTIONS:
    #   period     → write data every N timesteps (integer)
    #   dt_period  → write data every Δt seconds of physical time
    #   Only one of the two may be specified.
    #
    # ITERATION RANGE:
    #   iteration_min → first iteration to write (inclusive)
    #   iteration_max → last iteration to write (exclusive)
    #
    # EXAMPLE:
    #   iteration_min = 0, iteration_max = 2000
    #   period = 100   → diagnostic writes at: 0, 100, 200, ..., 1900
    # --------------------------------------------------------------------------

    diag = SynchrotronRadiationDiagnostic(
        period        = diag_period,         # Write every diag_period timesteps
        # dt_period   = None,                # (Alternative) Write every Δt seconds - Leave as None if using "period".

        species       = dic_species,         # Dict of species that emit and radiation is recorded.
        				     # Each species must have synchrotron radiation enabled
                                             # The key becomes the name in output, like
						     # species_dict = {
						     # "electrons": elec,
						     # "positrons": posit
						     # }
					     # IMPORTANT - with identical spectral–angular grids.

       # write_dir     = 'diags_synch',       # Directory where openPMD radiation files are stored
       					      # Defaults to current working directory if omitted.

        comm          = sim.comm,            # REQUIRED for parallel runs:
                                             #   - gathers data on rank 0
                                             #   - removes guard cells
                                             # If None: each MPI rank writes its own file


		                             
                            
        iteration_min = 0,                   # Start writing from iteration 0 (inclusive)
        iteration_max = np.inf               # Continue until simulation ends
    )

    # Attach diagnostic to the simulation
    sim.diags.append(diag)
