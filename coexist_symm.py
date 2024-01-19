from packages import*
from numerical_param import*
import selfe_bulk
import calculate
from scipy import integrate

def binodal(concns,valency,rad_ions,vol_sol,epsilon):
    G =np.zeros(len(valency))
    J= np.zeros((len(valency),len(valency)))
    guess = np.copy(concns)
    guess1 = np.copy(guess)
    convergence = np.inf
    p=0
    while (convergence > tolerance_phases):
        x = guess[0]
        y = guess[1]
        n1 = np.array([x,x])
        n2 = np.array([y,y])
        kappa1 = calculate.kappa_loc(n1, valency, epsilon)
        kappa2 = calculate.kappa_loc(n2, valency, epsilon)
        u1 = selfe_bulk.uself_short_single(n1, rad_ions, valency, epsilon)
        u2 = selfe_bulk.uself_short_single(n2, rad_ions, valency, epsilon)

        I1 = np.zeros(len(valency))
        I2 = np.zeros(len(valency))
        for i in range(0,len(valency)):
            I1[i] = integrate.quad(selfe_bulk.uself_bulk_charge, 0, 1, args=(kappa1, rad_ions[i], valency[i], epsilon))[0] - u1[i]
            I2[i] = integrate.quad(selfe_bulk.uself_bulk_charge, 0, 1, args=(kappa2, rad_ions[i], valency[i], epsilon))[0] - u2[i]
        G[0] = u1[0] - u2[0] + log(x) - log(y) + vol_sol*(x*np.sum(I1) - y*np.sum(I2))
        G[1] = log(1-2*vol_sol*x) - log(1-2*vol_sol*y) + vol_sol*(x*np.sum(I1) - y*np.sum(I2))

        DItau1 = np.zeros(len(valency))
        DI1 = np.zeros(len(valency))
        DItau2 = np.zeros(len(valency))
        DI2 = np.zeros(len(valency))
        for i in range(0,len(valency)):
            DI1[i] = selfe_bulk.duself_gauss(1,kappa1,rad_ions[i],valency[i],epsilon)
            DI2[i] = selfe_bulk.duself_gauss(1,kappa2,rad_ions[i],valency[i],epsilon)
            DItau1[i] = n1[i]* integrate.quad(selfe_bulk.duself_gauss, 0, 1, args=(kappa1, rad_ions[i], valency[i], epsilon))[0] - n1[i]*DI1[i]
            DItau2[i] = n2[i]* integrate.quad(selfe_bulk.duself_gauss, 0, 1, args=(kappa2, rad_ions[i], valency[i], epsilon))[0] - n2[i]*DI2[i]

        c1 = valency[0]**2/(kappa1*epsilon)
        c2 = valency[0]**2/(kappa2*epsilon)
        J[0,0] = 1/x + c1*DI1[0] + vol_sol*(np.sum(I1) + c1*np.sum(DItau1))
        J[0,1] = -1/y - c2*DI2[0] - vol_sol*(np.sum(I2) + c2*np.sum(DItau2))
        J[1,0] = (-2*vol_sol)/(1-2*x*vol_sol) + vol_sol*(np.sum(I1) + c1*np.sum(DItau1))
        J[1,1] = (2*vol_sol)/(1-2*y*vol_sol) - vol_sol*(np.sum(I2) + c2*np.sum(DItau2))

        dguess = np.linalg.solve(J, -G)
        guess1 = guess + dguess
        convergence = max(abs(np.true_divide(dguess,guess)))
        guess = phase_ratio*guess1 + (1-phase_ratio)*guess
        if p%1000==0:
            print(convergence)
        #print(np.linalg.norm(G))
        #print(np.linalg.norm(dguess))
        p = p+1
    n_bulk1 = np.array([guess1[0],guess1[0]])
    n_bulk2 = np.array([guess1[1],guess1[1]])
    return n_bulk1,n_bulk2
