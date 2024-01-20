from packages import *
import num_concn
import calculate
import selfe_vap_liq
import selfe_bulk
from numerical_param import*

def mgrf_vap_liq(psi_guess,nconc_guess,n_bulk1,n_bulk2,valency,rad_ions,vol_ions,vol_sol,domain, epsilon):
    
    print('selfe_ratio= ' + str(selfe_ratio))
    grid_points = len(psi_guess)
    bounds = (0,domain)
    Lz = bounds[1]
    n_bulk = n_bulk1
    psi_G = psi_guess[-1]
    psi_g = np.copy(psi_guess)
    eta_profile=calculate.eta_profile(nconc_guess,vol_ions,vol_sol)
    uself_profile = selfe_vap_liq.uself_complete(nconc_guess,n_bulk1,n_bulk2,rad_ions,valency,domain,epsilon)
    uself = np.copy(uself_profile)
    print('selfe_done before the loop')

    # Bulk properties
    n_bulk_numerical = np.multiply(np.ones((grid_points,len(valency))),n_bulk)
    uself_bulk = selfe_bulk.uselfb_numerical(n_bulk_numerical, n_bulk, rad_ions, valency, domain,epsilon)[-1]
    eta_bulk = calculate.eta_loc(n_bulk, vol_ions, vol_sol)

    # Checking if all molecules have same excluded volume
    vol_diff = np.abs(vol_ions - vol_sol)
    equal_vols = np.all(vol_diff < vol_sol * 1e-5)

    n_profile, coeffs = num_concn.nconc_mgrf(psi_g, uself_profile, eta_profile, uself_bulk, n_bulk, valency, vol_ions, eta_bulk, equal_vols)
    coeffs = coeffs/epsilon

    Z = None

    # Solving the matrix
    convergence_tot = np.inf
    p=1
    while(convergence_tot > tolerance_mgrf):

        # Bases
        coords = d3.CartesianCoordinates('z')
        dist = d3.Distributor(coords, dtype=np.float64)  # No mesh for serial / automatic parallelization
        zbasis = d3.Chebyshev(coords['z'], size=grid_points, bounds=bounds, dealias=dealias)

        # Fields
        z = dist.local_grids(zbasis)
        psi = dist.Field(name='psi', bases=zbasis)
        tau_1 = dist.Field(name='tau_1')  # the basis here is the edge
        tau_2 = dist.Field(name='tau_2')  # the basis here is the edge

        # Substitutions
        dz = lambda A: d3.Differentiate(A, coords['z'])
        lift_basis = zbasis.derivative_basis(2)
        lift = lambda A, n: d3.Lift(A, lift_basis, n)
        c0 = dist.Field(bases = zbasis)
        c1 = dist.Field(bases = zbasis)
        n_profile_useless, coeffs = num_concn.nconc_mgrf(psi_g, uself, eta_profile, uself_bulk, n_bulk, valency, vol_ions,eta_bulk, equal_vols)
        coeffs = coeffs/epsilon

        # lambda function for RHS, dedalus understands lambda functions can differentiate it for newton iteration
        c0['g'] = np.squeeze(coeffs[:, 0])
        c1['g'] = np.squeeze(coeffs[:, 1])
        boltz0 = lambda psi: np.exp(-valency[0] * psi)
        boltz1 = lambda psi: np.exp(-valency[1] * psi)


        # PDE setup
        problem = d3.NLBVP([psi, tau_1, tau_2], namespace=locals())
        problem.add_equation("-lap(psi) + lift(tau_1,-1) + lift(tau_2,-2) = c0*boltz0(psi) + c1*boltz1(psi)")

        # Boundary conditions
        problem.add_equation("dz(psi)(z=0) = 0")
        #problem.add_equation("dz(psi)(z=Lz) = 0")
        problem.add_equation("(psi)(z=Lz) = psi_G")

        # Initial Guess
        psi['g'] = psi_g

        # Solver
        solver = problem.build_solver(ncc_cutoff=ncc_cutoff_mgrf)
        pert_norm = np.inf
        psi.change_scales(dealias)
        s = 0
        while pert_norm > tolerance_pb:
            solver.newton_iteration()
            pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
            s = s + 1

        psi.change_scales(1)
        psi_g = psi['g']
        print('PB done')
        n_profile,coeff_useless = num_concn.nconc_mgrf(psi_g, uself, eta_profile, uself_bulk, n_bulk, valency, vol_ions, eta_bulk,equal_vols)
        uself_profile = selfe_vap_liq.uself_complete(n_profile, n_bulk1,n_bulk2,rad_ions, valency, domain,epsilon)
        #print('selfe_done in the loop')
        convergence_tot = np.true_divide(np.linalg.norm(uself_profile - uself),np.linalg.norm(uself))

        # mixing old self-energy and new self-energy
        uself = selfe_ratio*uself_profile + (1-selfe_ratio)*uself

        # mixing old eta and new_eta
        eta_profile = eta_ratio*calculate.eta_profile(n_profile,vol_ions,vol_sol) +(1-eta_ratio)*eta_profile
        q_profile = calculate.charge_density(n_profile, valency)

        Z = np.squeeze(z)
        
        # deleting dedalus fields as precuation
        del coords,dist,zbasis,z,psi,tau_1,tau_2,dz,lift_basis,lift,problem,solver,pert_norm,c0,c1,boltz0,boltz1

        p = p+1
        if p%1==0:
            print('converg at iter = ' + str(p) + ' is ' + str(convergence_tot))

    q_profile = calculate.charge_density(n_profile, valency)
    res= calculate.res_vap_liq(psi_g,q_profile,bounds,epsilon) + np.linalg.norm(n_profile[0] + n_profile[-1] - n_bulk1 - n_bulk2)
    print("Gauss's law residual for MGRF is is = " + str(res))

    return psi_g, n_profile,uself_profile,q_profile,Z, res

 