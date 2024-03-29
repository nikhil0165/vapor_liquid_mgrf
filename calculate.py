import numpy as np

from packages import *
from numerical_param import*

def ionic_strength(n_position,valency):  # n_position corresponds to number of ions for a particular position/node
    I = np.sum((1 / len(valency)) * (np.power(valency,2)) * n_position)
    return I

def charge(psi, valency, n_bulk): # charge at any position for mean-field PB
    return valency * n_bulk * exp(-valency * psi)

def kron_delta(i, j): # implementation of kronecker delta
    return 1 if i == j else 0

def charge_density(n_profile,valency): # charge density in the entire domain
    q_profile = np.dot(n_profile, valency)
    return q_profile

def eta_loc( n_position,vol_ions, vol_sol): # eta factor at any local position
    vol_local = np.sum(vol_ions * n_position)
    return (-1 / vol_sol) * log(1 - vol_local)

def eta_profile(n_profile,vol_ions, vol_sol): # eta factor at all locations
    eta_profile = np.apply_along_axis(eta_loc, 1, n_profile, vol_ions, vol_sol)
    return eta_profile

def kappa_loc(n_position,valency,epsilon):  # screening factor at a particular position
    I = np.sum((1 / len(valency)) * (np.power(valency,2)) * n_position)
    kappa = sqrt(I * (len(valency) / epsilon))
    return kappa

def kappa_sqr(n_position,valency, epsilon):  # square of screening factor at a particular position
    I = np.sum((1 / len(valency)) * (np.power(valency,2)) * n_position)
    return (I * (len(valency) / epsilon))

def kappa_sqr_profile(n_profile,valency,epsilon): #square of screening factor for all positions
    I = np.sum((1 / len(valency)) * (np.power(valency,2)) * n_profile, axis=1)
    return (I * (len(valency) / epsilon))

def kappa_profile(n_profile,valency,  epsilon): #screening factor for all positions
    kappa = np.apply_along_axis(kappa_loc,1,n_profile,valency,epsilon)
    return kappa

def interpolator(psi_profile,domain,points):

    grid_points = len(psi_profile)
    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = (0,domain))

    psi = dist.Field(name = 'psi',bases = zbasis)
    psi['g'] = psi_profile

    psi_answer = np.zeros(len(points))
    for i in range(0,len(points)):
        psi_answer[i] = psi(z = points[i]).evaluate()['g'][0]

    return psi_answer

def rescaler(psi_profile,n_profile,bounds,new_grid): # function to change grid points of psi and nconc fields

    grid_points = len(psi_profile)
    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = grid_points,bounds = bounds)

    # Fields
    n_ions = len(n_profile[0,:])
    nconc = np.zeros((new_grid,n_ions))
    psi = dist.Field(name = 'psi',bases = zbasis)
    psi['g'] = psi_profile
    psi.change_scales(new_grid/grid_points)

    nconc0 = dist.Field(name = 'nconc0',bases = zbasis)
    nconc1 = dist.Field(name = 'nconc1',bases = zbasis)
    nconc0['g'] = n_profile[:,0]
    nconc1['g'] = n_profile[:,1]
    nconc0.change_scales(new_grid/grid_points)
    nconc1.change_scales(new_grid/grid_points)
    nconc[:,0] = nconc0.allgather_data('g')
    nconc[:,1] = nconc1.allgather_data('g')
    if n_ions==4:
        nconc2 = dist.Field(name = 'nconc2',bases = zbasis)
        nconc3 = dist.Field(name = 'nconc3',bases = zbasis)
        nconc2['g'] = n_profile[:,2]
        nconc3['g'] = n_profile[:,3]
        nconc2.change_scales(new_grid/grid_points)
        nconc3.change_scales(new_grid/grid_points)
        nconc[:,2] = nconc2.allgather_data('g')
        nconc[:,3] = nconc3.allgather_data('g')

    return psi.allgather_data('g'), nconc



def res_asymm(psi_profile,n_profile,n_bulk1,n_bulk2,psi2,valency,bounds,epsilon): # calculate the residual of gauss law

    nodes = len(psi_profile)
    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype = np.float64)  # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'],size = nodes,bounds = bounds,dealias = dealias)
    q_profile = charge_density(n_profile, valency)

    # Fields
    z = dist.local_grids(zbasis)
    psi = dist.Field(name = 'psi',bases = zbasis)
    psi['g'] = psi_profile

    grad_psi = d3.Differentiate(psi,coords['z'])
    lap_psi = d3.Laplacian(psi).evaluate()
    lap_psi.change_scales(1)

    slope_0 = grad_psi(z = 0).evaluate()['g'][0]
    slope_end = grad_psi(z = bounds[1]).evaluate()['g'][0]
    res = np.zeros(nodes+2)
    
    res[0] = psi(z = 0).evaluate()['g'][0]
    res[nodes-1] = psi(z = bounds[1]).evaluate()['g'][0] - psi2
    res[1:nodes-1] = lap_psi['g'][1:nodes-1] + q_profile[1:nodes-1]/epsilon
    res[nodes] = np.linalg.norm(n_profile[0] - n_bulk1)
    res[nodes+1] = np.linalg.norm(n_profile[-1] - n_bulk2)
    
    return np.max(np.abs(res)), psi(z = 0.5*bounds[1]).evaluate()['g'][0]

