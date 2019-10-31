## @mainpage
## @section intro_sec Introduction
## This is a Doxygen-generated documentation page for the
## dmdpaper github repo at https://github.com/akaptano/dmdpaper.
##
## This repository is for the DMD paper published by A. A. Kaptanoglu et al.
##
## Compatibility requires installation of Python 3.7 and the typical
## python packages like scipy, numpy, matplotlib, and numba
## This code is intended to be compatible with Windows, MacOS, and Linux.

## @package psitet
## Defines the psi-tet dictionary and the routine
## to compare all the bulk quantities for a list
## of psi-tet dictionaries
from dataclasses import dataclass
## @class psiObject
## A data class which is used as a psi-tet dictionary
## All data sources are converted to this format.
## Note that new dictionary keys are added in various functions
## which are not listed here.
@dataclass(init=False, repr=True, eq=True)
class psiObject(object):
    ## Volumetric viscous heating
    visc = 0
    ## Volumetric electron advective heating
    e_adv = 0
    ## Volumetric electron conductive heating
    econd = 0
    ## Volumetric electron kinetic energy
    eke = 0
    ## Volumetric electron-ion collisional heating
    equil = 0
    ## Electron heat flux to the wall
    ewall = 0
    ## Volumetric field power
    fpow = 0
    ## Volumetric ion advective heating
    i_adv = 0
    ## Volumetric ion conductive heating
    icond = 0
    ## Ion heat flux to the wall
    iwall = 0
    ## Linear Iterations of the solver (psi-tet only)
    lits = 0
    ## Volume-averaged plasma density
    ne = 0
    ## Nonlinear Iterations of the solver (psi-tet only)
    nlits = 0
    ## Volumetric ohmic heating
    ohmic = 0
    ## Volumetric particle power (change in kinetic energy)
    ppow = 0
    ## Toroidal current
    tcurr = 0
    ## Volume-averaged electron temperature
    te = 0
    ## Toroidal flux
    tflux = 0
    ## Collisional heating
    therm = 0
    ## Volume-averaged ion temperature
    ti = 0
    ## Time base for xmhd.mat
    time = 0
    ## Injector power obtained from power balance
    inj_power = 0
    # hitn_driver.mat
    ## Injector current of the x-inj
    curr01 = 0
    ## Injector current of the y-inj
    curr02 = 0
    ## Injector flux of the x-inj
    flux01 = 0
    ## Injector flux of the y-inj
    flux02 = 0
    ## Time base for hitn_driver.mat
    driver_time = 0
    # heat_flux.mat
    ## Electron heat flux at every point in the volume
    e_flux = 0
    ## Ion heat flux at every point in the volume
    i_flux = 0
    ## Time base for heat_flux.mat
    heat_flux_time = 0
    # sp_probes.mat
    ## Bx for the functional surface probes
    sp_Bx = 0
    ## By for the functional surface probes
    sp_By = 0
    ## Bz field for the functional surface probes
    sp_Bz = 0
    ## Poloidal B field for the functional surface probes
    sp_Bpol = 0
    ## Toroidal B field for the functional surface probes
    sp_Btor = 0
    ## Time base for the functional surface probes
    sp_time = 0
    ## Names for the functional surface probes
    sp_names = '0'
    # imp_probes.mat
    ## Magnitude of B for the functional imp probes
    imp_B = 0
    ## Bx for the functional imp probes
    imp_Bx = 0
    ## By for the functional imp probes
    imp_By = 0
    ## Bz for the functional imp probes
    imp_Bz = 0
    ## Poloidal B for the functional imp probes
    imp_Bpol = 0
    ## Toroidal B for the functional imp probes
    imp_Btor = 0
    ## Radial B for the functional imp probes
    imp_Brad = 0
    ## Time base for the functional imp probes
    imp_time = 0
    # inter_probes.mat
    ## FIR density signal (chord-averaged)
    inter_n = 0
    ## Time base for the FIR signal
    inter_time = 0
    # idsn_probes.mat
    ## Raw IDS density signal
    ids_n = 0
    ## Time base for raw IDS density signal
    ids_ntime = 0
    # idsT_probes.mat
    ## Raw IDS Temperature Signal
    ids_T = 0
    ## Time base for raw IDS (Ion) Temperature signal
    ids_Ttime = 0
    # idsV_probes.mat
    ## Raw IDS Velocity Signal
    ids_V = 0
    ## Time base for raw IDS (Ion) Velocity signal
    ids_Vtime = 0
    # thomson_probes.mat
    ## Thomson measurements of electron temperature
    thomson_te = 0
    ## Time base for Thomson measurements
    thomson_time = 0
    ## start time for analysis and plotting
    t0 = 0
    ## end time for analysis and plotting
    tf = 0
    ## Injector frequency
    freq = 0
    ## Flag to indicate if this is a HITSI3 dictionary
    is_HITSI3 = 0
