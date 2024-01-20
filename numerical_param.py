from packages import *

## Numerical Parameters

s_conv = 1e4  # approx for infinity for fourier inverse of greens function
V_conv = log(s_conv + 1)  # we do fourier inverse integration in the logspace
quads = 20  # no of legendre gauss quadrature points for fourier inverse of greens function
N_grid = 512 # has to be even, since we often use 3/2 for dealiasing
dealias = 3/2  # dealiasing factor for dedalus
ncc_cutoff_mgrf = 1e-3 # some cutoff parameter for non-constant coefficients on LHS of NLBVP of MGRF
ncc_cutoff_greens = 1e-1 # some cutoff parameter for non-constant coefficients on LHS of NLBVP of G
num_ratio = 0.1 # mixing ratio of new to old in nconc_mgrf
selfe_ratio = 0.1 # mixing ratio of self-energy (new to old) in outermost loop of pb_mgrf
eta_ratio = 0.1  # mixing ratio of eta (new to old) in outermost loop of pb_mgrf
phase_ratio = 0.5 # mixing ratio to calculate binodal curve
grandfe_quads = 50  # no of legendre gauss quadrature points for free energy calculation
cores = 10 # no of parallel processes in which you want to divide fourier inverse calculation
tolerance_mgrf = 1.0 * pow(10,-7)  # tolerance_mgrf for outermost loop for pb_mgrf
tolerance_pb = 1.0 * pow(10,-7)  # tolerance_pb for inner loop in mgrf_vap_liq
tolerance_num = pow(10,-5)  # tolerance_mgrf for nconc_mgrf iteration loop
tolerance_greens = pow(10,-5)  # tolerance_mgrf for nonlinear problem for greens function
tolerance_phases = pow(10,-6)# tolerance while calculating non-linear equations for two bulk phases
iter_max = pow(10,7)  # maximum no of iterations for any  iteration loop