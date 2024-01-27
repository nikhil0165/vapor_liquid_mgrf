from packages import *
from numerical_param import *
import calculate
import selfe_vap_liq
import selfe_bulk
import poisson_interface

# function to calculate concn profiles for mean-field PB
def nconc_pb(psi, valency, n_bulk):
    return n_bulk * np.exp(-np.array(valency) * psi[:,np.newaxis])

def nguess_symm(n_bulk1,n_bulk2,valency,int_width, epsilon,grid_points):
    lambda1= 1/calculate.kappa_loc(n_bulk1,valency,epsilon)

    p = (n_bulk2-n_bulk1)/2
    q = (n_bulk2+n_bulk1)/2

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = (0,2*int_width*lambda1))
    z = np.squeeze(dist.local_grids(zbasis))

    n_guess = np.multiply(p[:,np.newaxis],np.tanh((z - int_width * lambda1) / lambda1)) + q[:,np.newaxis]
    n_guess = n_guess.T
    return n_guess, 2*int_width*lambda1

# def nguess_asymm(n_bulk1,n_bulk2,psi2,valency,int_width, epsilon,grid_points):
#     lambda1= 1/calculate.kappa_loc(n_bulk1,valency,epsilon)
#
#     p = (n_bulk2-n_bulk1)/2
#     q = (n_bulk2+n_bulk1)/2
#
#     coords = d3.CartesianCoordinates('z')
#     dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
#     zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = (0,2*int_width*lambda1))
#     z = np.squeeze(dist.local_grids(zbasis))
#
#     n_guess = np.multiply(p[:,np.newaxis],np.tanh((z - int_width* lambda1) / lambda1)) + q[:,np.newaxis]
#     n_guess = n_guess.T
#
#     psi_guess = 0.5*psi2*np.tanh((z - int_width* lambda1) / lambda1) + 0.5*psi2
#     return n_guess, psi_guess, 2*int_width*lambda1

def nguess_asymm(n_bulk1,n_bulk2,psi2,valency,int_width, epsilon,grid_points):

    lambda1= 1/calculate.kappa_loc(n_bulk1,valency,epsilon)
    p = (n_bulk2[0]-n_bulk1[0])/2
    q = (n_bulk2[0]+n_bulk1[0])/2

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = (0,3*int_width*lambda1),dealias=3/2)
    z = np.squeeze(dist.local_grids(zbasis))

    psi_guess = 0.5*psi2*np.tanh((z - 2*int_width* lambda1) / lambda1) + 0.5*psi2

    psi = dist.Field(name = 'psi',bases = zbasis)
    psi['g'] = psi_guess

    lap_psi = d3.Laplacian(psi).evaluate()
    lap_psi.change_scales(1)

    q_profile = -lap_psi['g']*epsilon # Gauss law

    nconc0 = p*np.tanh((z - int_width* lambda1) / lambda1) + q
    nconc1 = (q_profile - valency[0]*nconc0)/valency[1]

    n_guess= np.c_[nconc0,nconc1]
    return n_guess, psi_guess, 2*int_width*lambda1, z

# function to calculate num and coeffs for mgrf_vap_liq use
def nconc_mgrf(psi,uself,eta_profile,uself_bulk, n_bulk, valency, vol_ions,psi_bulk,eta_bulk, equal_vols):
    if equal_vols:
        A = n_bulk* np.exp(-np.array(valency) * (psi[:,np.newaxis]-psi_bulk) - (uself - uself_bulk) + vol_ions * eta_bulk)
        coeffs = valency * n_bulk* np.exp(np.array(valency)*psi_bulk - (uself - uself_bulk) + vol_ions*eta_bulk)
        denom = 1 + np.sum(A * vol_ions, axis=1)
        n_profile= np.true_divide(A,denom[:,np.newaxis])
        coeffs = np.true_divide(coeffs,denom[:,np.newaxis])
    else:
        n_profile = n_bulk * np.exp(-np.array(valency)*(psi[:,np.newaxis]-psi_bulk) - (uself - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        coeffs = valency* n_bulk * np.exp(np.array(valency)*psi_bulk -(uself - uself_bulk) - vol_ions* (eta_profile[:,np.newaxis] - eta_bulk))
    return n_profile,coeffs

# function to calculate concentration profile for given psi profile and bulk conditions, n_initial is the initial guess


# function to calculate concentration profile for given psi profile and bulk conditions, n_initial is the initial guess
def nconc_asymm(psi_initial, n_initial,n_bulk1,n_bulk2, psi2, valency, rad_ions, vol_ions, vol_sol, domain, epsilon):

    n_bulk = n_bulk2 # choose any one the bulk phases.
    eta_bulk = calculate.eta_loc(n_bulk, vol_ions, vol_sol)
    psi_bulk = psi2
    nodes = len(n_initial)
    bounds = (0,domain)

    n_bulk_profile = np.multiply(np.ones((nodes,len(valency))),n_bulk)
    uself_bulk = selfe_bulk.uselfb_numerical(n_bulk_profile,n_bulk,rad_ions, valency,domain,epsilon)
    print('bulk selfe done')
    equal_vols = np.all(np.abs(vol_ions - vol_sol) < vol_sol * 1e-5)
    # profile variables
    n_profile = np.copy(n_initial)
    n_guess = np.copy(n_initial)
    psi_guess = np.copy(psi_initial)

    # initializing the self energy 
    uself_profile = selfe_vap_liq.uself_complete(n_guess, n_bulk1,n_bulk2,rad_ions, valency,domain, epsilon)
    eta_profile = calculate.eta_profile(n_initial,vol_ions,vol_sol)

    # Iteration
    convergence = np.inf
    p = 0
    while (convergence > tolerance_num) and (p < iter_max):
        psi_guess = poisson_interface.poisson_interface(n_guess,valency,psi2,domain,epsilon)
        p = p + 1
        if equal_vols:
            A = n_bulk* np.exp(-np.array(valency) * (psi_guess[:, np.newaxis]-psi_bulk) - (uself_profile - uself_bulk) + vol_ions * eta_bulk)
            denom = 1 + np.sum(A * vol_ions, axis=1)
            n_profile = np.true_divide(A, denom[:,np.newaxis])
        else:
            n_profile = n_bulk * np.exp(-np.array(valency)*(psi_guess[:,np.newaxis]-psi_bulk) - (uself_profile - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        print(np.any(np.isnan(n_profile)))
        convergence = np.true_divide(np.linalg.norm(n_profile - n_guess),np.linalg.norm(n_guess))
        n_guess = (num_ratio) * n_profile + (1-num_ratio) * n_guess
        uself_profile = selfe_vap_liq.uself_complete(n_guess,n_bulk1,n_bulk2, rad_ions, valency, domain,epsilon)
        eta_profile = calculate.eta_profile(n_guess,vol_ions,vol_sol)
        #print(np.any(np.isnan(psi_guess)))
        if p%10==0:
            print('converg at iter = ' + str(p) + ' is ' + str(convergence))
        if p >= iter_max:
            print("too many iterations for convergence")
    q_profile = calculate.charge_density(n_guess,valency)
    res = np.linalg.norm(q_profile) + np.linalg.norm(n_profile[0] + n_profile[-1] - n_bulk1 - n_bulk2)

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = nodes,bounds = bounds)
    z = np.squeeze(dist.local_grids(zbasis))

    return psi_guess,n_profile, uself_profile,q_profile,z,res