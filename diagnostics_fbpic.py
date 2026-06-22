# diagnostics.py

from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
from fbpic.openpmd_diag import ParticleChargeDensityDiagnostic

# --------------------------------------
# Diagnostics
# --------------------------------------
# Field Diagnostic
# ----------------
# This diagnostic saves the electromagnetic fields (E, B), charge density
# (rho), and current density (J) to openPMD files every diag_period steps.
# These outputs are essential for analyzing:
  
# The diagnostic reads the field object sim.fld and writes it to the
# standard diagnostics directory (diags/hdf5), unless another write_dir is specified. 
#The comm option ensures proper MPI communication in multi‑processor runs.

def field_diags(sim, diag_period):
    diag_field = FieldDiagnostic(
        period     = diag_period,   # how often to write field data
    fldobject  = sim.fld,       # field solver’s field container
    comm       = sim.comm       # MPI communicator (handles guard-cell removal)
    )
    
    return diag_field


# -----------------------------------
# Particle Diagnostic (Electron Beam)
# -----------------------------------
    
# This diagnostic saves particle phase‑space data for the electron species.
#
# What it records:
#   • Positions (x, y, z)
#   • Momenta (ux, uy, uz = p/(mc))
#   • Weights (w), particle IDs (if enabled)
#
# The `select` option filters the saved particles:    
#   select = {"uz": [1., None]}
#     → Only particles with uz ≥ 1 are written.
#     → uz = pz/(mc), so uz ≥ 1 corresponds to *relativistic forward‑moving electrons.
#
#
# The diagnostic writes openPMD particle data into the same directory as
# the field diagnostics (unless write_dir is overridden).
def particle_diags(sim, diag_period, species):
    diag_particle = ParticleDiagnostic(
        period  = diag_period,          # how often to save particle data
        species = species,  # which particle species to dump
        comm    = sim.comm              # MPI communicator for parallel runs
    )

    return diag_particle


def particle_charge_density_diags(
        sim,
        diag_period,             # timestep period (int) OR dt_period (float seconds)
        species_dict,            # dict of species: {'driver': driver, 'plasma': plasma_e, ...}
        write_dir=None,        # directory where results go (None = CWD)
        iteration_min=0,              # first iteration to record (inclusive)
        iteration_max=float('inf')    # last iteration to record (exclusive)
    ):
    """
    Create a ParticleChargeDensityDiagnostic with full-parameter control.

    Parameters
    ----------
    sim : fbpic.Simulation
        The active simulation object.

    diag_period : int or float or None
        If int   -> interpreted as 'period' (write every N time steps).
        If float -> interpreted as 'dt_period' (write every Δt seconds).
        If None  -> no writing cadence specified (not recommended).

    species_dict : dict
        Dictionary of species to diagnose,
        e.g. {'driver': driver_species, 'plasma': plasma_electrons}.
        Keys are the names used in the openPMD output.

    write_dir : str or None
        Where to write the diagnostic output.
        None -> writes into the current working directory.

    iteration_min : int
        First iteration at which to start writing diagnostics (inclusive).

    iteration_max : int
        Last iteration before which writing stops (exclusive).

    Returns
    -------
    diag_rho : ParticleChargeDensityDiagnostic
        The initialized diagnostic object.
    """

    

    # ------------------------------------------------------------
    # Determine whether diag_period is period or dt_period
    # ------------------------------------------------------------
    if isinstance(diag_period, int):
        period = diag_period
        dt_period = None
    elif isinstance(diag_period, float):
        period = None
        dt_period = diag_period
    else:
        raise ValueError(
            "diag_period must be int (timestep) or float (seconds)."
        )

    # ------------------------------------------------------------
    # Create the diagnostic
    # ------------------------------------------------------------
    diag_rho = ParticleChargeDensityDiagnostic(
        period        = period,          # timestep period
        dt_period     = dt_period,       # physical time period
        sim           = sim,             # simulation object
        species       = species_dict,    # dict of species
        write_dir     = write_dir,       # output directory
        iteration_min = iteration_min,   # start iteration
        iteration_max = iteration_max    # end iteration
    )

    return diag_rho