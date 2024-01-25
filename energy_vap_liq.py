from packages import *
from numerical_param import *
import selfe_vap_liq
import selfe_bulk

# free energy from mgrf theory for vap_liquid interface
def grandfe_mgrf_vap_liq(psi, n_profile, uself_profile,n_bulk1,n_bulk2, psi2 ,valency,rad_ions, vol_ions, vol_sol, domain, epsilon):

    nodes = len(n_profile)-1
    n_bulk = n_bulk2
    psi_bulk = psi2 # if n_bulk = n_bulk2 then
    n_bulk_profile = np.multiply(np.ones((nodes+1, len(valency))), n_bulk2)
    grandfe_bulk = grandfe_mgrf_bulk(n_bulk_profile,n_bulk, psi_bulk, valency,rad_ions, vol_ions,vol_sol, domain, epsilon)
    utau = np.zeros_like(uself_profile)
    taus, weights = np.polynomial.legendre.leggauss(grandfe_quads)

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords, dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'], size = len(n_profile), bounds = (0,domain))
    z = np.squeeze(dist.local_grids(zbasis))
    dz = np.diff(z)
    n_local = 0.5 * (n_profile[:-1] + n_profile[1:])
    psi_local = 0.5 * (psi[:-1] + psi[1:])
    u_local = 0.5 * (uself_profile[:-1] + uself_profile[1:])
    vol_local = np.sum(vol_ions * n_local, axis=1)

    grandfe = 0
    grandfe = grandfe - 0.5 * np.sum(psi_local * np.dot(valency, n_local.T) * dz)
    grandfe = grandfe - np.sum(n_local*dz[:,np.newaxis])
    grandfe = grandfe - (1 / vol_sol) * np.sum((1 - vol_local) * dz)
    grandfe = grandfe + (1 / vol_sol) * np.sum(np.log(1 - vol_local) * dz)

    for k in range(0, len(taus)):
        utau = utau + 0.5*weights[k]*selfe_vap_liq.uself_complete((0.5*taus[k]+0.5)*n_profile,(0.5*taus[k]+0.5)*n_bulk1,(0.5*taus[k]+0.5)*n_bulk2,rad_ions, valency,domain, epsilon)

    utau_local = 0.5 * (utau[:-1] + utau[1:])
    grandfe = grandfe + np.sum(n_local * utau_local * dz[:, np.newaxis])
    grandfe = grandfe - np.sum(n_local * u_local * dz[:, np.newaxis])
    return grandfe - grandfe_bulk

# free energy from mgrf theory for bulk solution

def grandfe_mgrf_bulk(n_bulk_profile,n_bulk, psi_bulk, valency,rad_ions,vol_ions, vol_sol, domain, epsilon):

    nodes = len(n_bulk_profile)
    vol_bulk = sum([n_bulk[i] * vol_ions[i] for i in range(len(vol_ions))])
    u_bulk = selfe_bulk.uselfb_numerical(n_bulk_profile, n_bulk, rad_ions, valency, domain, epsilon)


    utau_bulk = np.zeros_like(u_bulk)
    taus, weights = np.polynomial.legendre.leggauss(grandfe_quads)

    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords, dtype = np.float64)
    zbasis = d3.Chebyshev(coords['z'], size = nodes+1, bounds = (0,domain))
    z = np.squeeze(dist.local_grids(zbasis))
    dz =np.diff(z)

    grandfe = 0
    grandfe = grandfe - 0.5*psi_bulk*np.sum(np.dot(valency,n_bulk_profile.T) * dz)
    grandfe = grandfe - np.sum(n_bulk * dz[:, np.newaxis])
    grandfe = grandfe - np.sum(dz*(1/vol_sol)*(1 - vol_bulk))
    grandfe = grandfe + np.sum(dz*(1/vol_sol)*np.log(1-vol_bulk))

    for k in range(0, len(taus)):
        utau_bulk = utau_bulk + 0.5 * weights[k] * selfe_bulk.uselfb_numerical((0.5 * taus[k] + 0.5) * n_bulk_profile,(0.5*taus[k]+0.5)*n_bulk, rad_ions, valency,domain, epsilon)

    grandfe = grandfe + np.sum(n_bulk * utau_bulk * dz[:,np.newaxis])
    grandfe = grandfe - np.sum(n_bulk * u_bulk * dz[:,np.newaxis])

    return grandfe


