from packages import *
import calculate
import num_concn

def poisson_interface(n_profile,valency,psi_G,domain,epsilon):

    bounds = (0,domain)
    Lz = bounds[1]
    grid_points = len(n_profile)
    # Bases
    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords, dtype=np.float64) # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'], size= grid_points,bounds =bounds)

    # Fields
    z = dist.local_grids(zbasis)
    psi = dist.Field(name='psi', bases=zbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    q_profile = dist.Field(name = 'q_profile',bases = zbasis)
    q_profile['g'] = calculate.charge_density(n_profile,valency)

    # Substitutions
    dz = lambda A: d3.Differentiate(A, coords['z'])
    lift_basis = zbasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    # PDE Setup
    problem = d3.LBVP([psi,tau_1, tau_2], namespace=locals())
    problem.add_equation("-lap(psi) + lift(tau_1,-1) + lift(tau_2,-2) = q_profile/epsilon")

    # Boundary conditions
    problem.add_equation("(psi)(z=0) = 0")
    problem.add_equation("(psi)(z=Lz) = psi_G")

    solver = problem.build_solver()
    solver.solve()

    return psi['g']
