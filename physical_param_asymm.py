from packages import *

## Global Input Variables, All quantities are in SI unit _d means dimensional
T_star= 0.07
T_star_in = 0.06
valency = np.array([2,-1]) # valency of primary salt
born_radius1 = 2.0
born_radius2 = 2.0
rad_sol_d = max(born_radius1,born_radius2)
int_width= 10.0 #  this times debye huckel length of phase 1
int_width_in  = int_width

print(f'valency: {valency}')
print(f'born_radius1 = {born_radius1}')
print(f'born_radius2 = {born_radius2}')
print(f'rad_sol_d = {rad_sol_d}')
print(f'T_star = {T_star}')
print(f'T_star_in = {T_star_in}')
print(f'int_width: {int_width}')
print(f'int_width_in: {int_width_in}')

vol_sol_d = 4/3*pi*pow(rad_sol_d,3)
rad_ions_d = np.array([born_radius1, born_radius2])
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
Temp_c  = abs(valency[0]*valency[1])*(ec**2)/(4*pi*epsilon_s_d*k_b*np.sqrt(born_radius1*born_radius2*pow(10,-20)))# shouldnt this be diameter also
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
rad_ions = np.true_divide(rad_ions_d* pow(10, -10),l_c)
rad_sol  = rad_sol_d* pow(10, -10)/l_c
vol_sol = vol_sol_d*pow(10,-30)/vol_c
vol_ions = np.true_divide(vol_ions_d*pow(10,-30),vol_c)
c_max = 1/vol_sol





