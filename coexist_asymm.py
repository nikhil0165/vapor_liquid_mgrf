from packages import*
from numerical_param import*
import selfe_bulk
import calculate
from scipy import integrate

def binodal(concns_psi,valency,rad_ions,v_sol,epsilon):
    G =np.zeros(3)
    J= np.zeros((3,3))
    guess = np.copy(concns_psi)
    guess1 = np.copy(guess)
    b = abs(valency[0])
    c = abs(valency[1])
    a = 1 + b
    ratio = 5e-2
    convergence = 1
    p=0
    J[0, 2] = -b
    J[1, 2] = 1
    J[2,2] = 0
    while (convergence > tolerance_phases) and (p < iter_max):
        x = guess[0]
        y = guess[1]
        z = guess[2]
        n1 = np.array([c*x,b*x])
        n2 = np.array([c*y,b*y])
        u1 = selfe_bulk.uself_short_single(n1, rad_ions, valency, epsilon)
        u2 = selfe_bulk.uself_short_single(n2, rad_ions, valency, epsilon)
        kappa1 = calculate.kappa_loc(n1, valency, epsilon)
        kappa2 = calculate.kappa_loc(n2, valency, epsilon)
        I1 = np.zeros(len(valency))
        I2 = np.zeros(len(valency))
        for i in range(0,len(valency)):
            I1[i] = n1[i]*(integrate.quad(selfe_bulk.uself_bulk_charge, 0, 1, args=(kappa1, rad_ions[i], valency[i], epsilon))[0] - u1[i])
            I2[i] = n2[i]*(integrate.quad(selfe_bulk.uself_bulk_charge, 0, 1, args=(kappa2, rad_ions[i], valency[i], epsilon))[0] - u2[i])

        G[0] = -valency[0]*z + u1[0] - u2[0] + log(x) - log(y) + v_sol*(np.sum(I1) - np.sum(I2))
        G[1] = -valency[1]*z + u1[1] - u2[1] + log(x) - log(y) + v_sol*(np.sum(I1) - np.sum(I2))
        G[2] = log(1-v_sol*x*a) - log(1-a*v_sol*y) + v_sol*(np.sum(I1) - np.sum(I2))

        DItau1 = np.zeros(len(valency))
        DI1 = np.zeros(len(valency))
        DItau2 = np.zeros(len(valency))
        DI2 = np.zeros(len(valency))
        for i in range(0,len(valency)):
            DI1[i] = selfe_bulk.duself_gauss(1,kappa1,rad_ions[i],valency[i],epsilon)
            DI2[i] = selfe_bulk.duself_gauss(1,kappa2,rad_ions[i],valency[i],epsilon)
            DItau1[i] = n1[i]* integrate.quad(selfe_bulk.duself_gauss, 0, 1, args=(kappa1, rad_ions[i], valency[i], epsilon))[0] - n1[i]*DI1[i]
            DItau2[i] = n2[i]* integrate.quad(selfe_bulk.duself_gauss, 0, 1, args=(kappa2, rad_ions[i], valency[i], epsilon))[0] - n2[i]*DI2[i]
        c1 = kappa1/(2*x)
        c2 = kappa2/(2*y)
        J[0,0] = 1/x + kappa1/(2*x)*DI1[0] + v_sol*(c1*np.sum(DItau1) + np.sum(I1)/x)
        J[0,1] = -(1/y + kappa2/(2*y)*DI2[0] + v_sol*(c2*np.sum(DItau2) + np.sum(I2)/y))
        J[1,0] = 1/x + kappa1/(2*x)*DI1[1] + v_sol*(c1*np.sum(DItau1) + np.sum(I1)/x)
        J[1,1] = -(1/y + kappa2/(2*y)*DI2[1] + v_sol*(c2*np.sum(DItau2) + np.sum(I2)/y))
        J[2,0] = -v_sol*a/(1-a*x*v_sol) + v_sol*(c1*np.sum(DItau1) + np.sum(I1)/x)
        J[2,1] = v_sol*a/(1-a*y*v_sol) - v_sol*(c2*np.sum(DItau2) + np.sum(I2)/y)

        dguess = np.linalg.solve(J, -G)
        guess1 = guess + dguess
        convergence = np.linalg.norm(dguess)/np.linalg.norm(guess)
        guess = ratio*guess1 + (1-ratio)*guess
        if p%1000==0:
            print('converg at iter = ' + str(p) + ' is ' + str(convergence))
        p = p+1

    n_bulk1 = np.array([ c*guess1[0],b *  guess1[0]])
    n_bulk2 = np.array([c*guess1[1],b * guess1[1]])
    return n_bulk1,n_bulk2, guess1[2]