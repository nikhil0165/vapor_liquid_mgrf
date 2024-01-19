import numpy as np

from packages import *
from numerical_param import *
import calculate
import selfe_vap_liq
import selfe_bulk

# function to calculate concn profiles for mean-field PB
def nconc_pb(psi, valency, n_bulk):
    return n_bulk * np.exp(-np.array(valency) * psi[:,np.newaxis] )

def nguess_tanh(n_bulk1,n_bulk2,valency,domain_1, domain_2, epsilon,grid_points):
    n_guess = np.zeros((grid_points,len(n_bulk1)),dtype=np.longdouble)
    lambda1= 1/calculate.kappa_loc(n_bulk1,valency,epsilon)
    lambda2 = 1/calculate.kappa_loc(n_bulk2,valency,epsilon)

    p = (n_bulk2-n_bulk1)/2
    q = (n_bulk2+n_bulk1)/2

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = (0,domain_1*lambda1 + domain_2*lambda2))
    z = np.squeeze(dist.local_grids(zbasis))

    n_guess = np.multiply(p[:,np.newaxis],np.tanh((z - domain_1 * lambda1) / lambda2)) + q[:,np.newaxis]
    n_guess = n_guess.T
    return n_guess, domain_1*lambda1 + domain_2*lambda2

# function to calculate num and coeffs for mgrf_vap_liq use
def nconc_mgrf(psi,uself,eta_profile,uself_bulk, n_bulk, valency, vol_ions,eta_bulk, equal_vols):
    if equal_vols:
        A = n_bulk* np.exp(-np.array(valency) * psi[:,np.newaxis] - (uself - uself_bulk) + vol_ions * eta_bulk)
        coeffs = valency * n_bulk* np.exp(-(uself - uself_bulk) + vol_ions * eta_bulk)
        denom = 1 + np.sum(A * vol_ions, axis=1)
        n_profile= np.true_divide(A,denom[:,np.newaxis])
        coeffs = np.true_divide(coeffs,denom[:,np.newaxis])
    else:
        n_profile = n_bulk * np.exp(-np.array(valency)*psi[:,np.newaxis] - (uself - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        coeffs = valency* n_bulk * np.exp(-(uself - uself_bulk) - vol_ions* (eta_profile[:,np.newaxis] - eta_bulk))
    return n_profile,coeffs

# function to calculate concentration profile for given psi profile and bulk conditions, n_initial is the initial guess
def nconc_complete(psi, n_initial,n_bulk1,n_bulk2, valency, rad_ions, vol_ions, vol_sol, domain, epsilon,equal_vols):

    n_bulk = n_bulk1
    eta_bulk = calculate.eta_loc(n_bulk, vol_ions, vol_sol)
    nodes = len(n_initial)
    bounds = (0,domain)

    n_bulk_profile = np.multiply(np.ones((nodes,len(valency))),n_bulk)
    uself_bulk = selfe_bulk.uselfb_numerical(n_bulk_profile,n_bulk1,rad_ions, valency,domain,epsilon)

    # profile variables
    n_profile = np.copy(n_initial)
    n_guess = np.copy(n_initial)

    # initializing the self energy 
    uself_profile = selfe_vap_liq.uself_complete(n_guess, n_bulk1,n_bulk2,rad_ions, valency,domain, epsilon)
    eta_profile = calculate.eta_profile(n_initial,vol_ions,vol_sol)

    # Iteration
    convergence = np.inf
    p = 0
    while (convergence > tolerance_num) and (p < iter_max):
        p = p + 1
        if equal_vols:
            A = n_bulk* np.exp(-np.array(valency) * psi[:, np.newaxis] - (uself_profile - uself_bulk) + vol_ions * eta_bulk)
            denom = 1 + np.sum(A * vol_ions, axis=1)
            n_profile = np.true_divide(A, denom[:,np.newaxis])
        else:
            n_profile = n_bulk * np.exp(-np.array(valency)*psi[:,np.newaxis] - (uself_profile - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        convergence = np.true_divide(np.linalg.norm(n_profile - n_guess),np.linalg.norm(n_guess))
        n_guess = (num_ratio) * n_profile + (1-num_ratio) * n_guess
        uself_profile = selfe_vap_liq.uself_complete(n_guess,n_bulk1,n_bulk2, rad_ions, valency, domain,epsilon)
        eta_profile = calculate.eta_profile(n_guess,vol_ions,vol_sol)
        if p%10==0:
            print('num='+str(convergence))
        if p >= iter_max:
            print("too many iterations for convergence")
    q_profile = calculate.charge_density(n_guess,valency)
    res = np.linalg.norm(q_profile) + np.linalg.norm(n_profile[0] + n_profile[-1] - n_bulk1 - n_bulk2)

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = nodes,bounds = bounds)
    z = np.squeeze(dist.local_grids(zbasis))


    return psi,n_profile, uself_profile,q_profile,z,res