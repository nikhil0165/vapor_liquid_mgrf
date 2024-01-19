from packages import *

## Global Input Variables, All quantities are in SI unit _d means dimensional
T_star= 0.05
T_star_in = 0.066
valency = np.array([2,-2]) # valency of primary salt
born_radius = 1.5* pow(10, -10)
rad_sol_d = born_radius
domain_1= 10.0 #  this times debye huckel length of phase 1
domain_2= 50.0 #  this times debye huckel length of phase 2
domain_1_in  = domain_1
domain_2_in = domain_2

print(f'valency: {valency}')
print(f'born_radius = {born_radius}')
print(f'rad_sol_d = {rad_sol_d}')
print(f'T_star = {T_star}')
print(f'domain_1: {domain_1}')
print(f'domain_2: {domain_2}')
print(f'T_star_in = {T_star_in}')
print(f'domain_1_in: {domain_1_in}')
print(f'domain_2_in: {domain_2_in}')

vol_sol_d = 4/3*pi*pow(rad_sol_d,3)
rad_ions_d = np.array([born_radius, born_radius])
vol_ions_d = np.array([vol_sol_d,vol_sol_d])
print(f'rad_ions_d: {rad_ions_d}')
print(f'vol_ions_d: {vol_ions_d}')
print(f'vol_sol_d: {vol_sol_d}')
## Physical constants

ec = 1.602 * pow(10, -19)  # electronic charge
k_b = 1.38064852 * pow(10, -23)  # boltzmann's constant
N_A = 6.02214*pow(10,23) # Avogadro number
epsilon_o_d = 8.854187 * pow(10, -12)  # permittivity in vaccuum
epsilonr_s_d = 80  # relative permittivity/dielectric constant
epsilon_s_d = epsilon_o_d * epsilonr_s_d  # permittivity of the medium
Temp_c  = abs(valency[0]*valency[1])*(ec**2)/(4*pi*epsilon_s_d*k_b*born_radius)# shouldnt this be diameter also
Temp = T_star*Temp_c
beta = 1 / (k_b * Temp)  # beta
print(f'Temp = {Temp}')## Characteristic variables - dont play with this

l_b = pow(ec,2)*(beta)*(1 / (4 * pi * epsilon_s_d)) #Bjerrum Length
l_c = l_b # characteristic length scale in the system
q_c = ec # characteristic charge
psi_c = (1/(beta*q_c))# characteristic electrostatic potential
epsilon_c = beta*q_c*q_c/l_c # characteristic dielectric permimitivity
sigma_c = (ec/pow(l_c,2)) # characteristic surface charge density
vol_c = pow(l_c,3) # characteristic volume
nconc_c = 1/vol_c # characteristic number density
conc_c = 1/vol_c # characteristic concentration


## Scaling the variables with characteristic variables

epsilon_s = epsilon_s_d / epsilon_c
rad_ions = np.true_divide(rad_ions_d,l_c)
rad_sol  = rad_sol_d/l_c
vol_sol = vol_sol_d/vol_c
vol_ions = np.true_divide(vol_ions_d,vol_c)
c_max = 1/vol_sol





