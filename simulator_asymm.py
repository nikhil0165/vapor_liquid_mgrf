from numerical_param import *
import num_concn
import energy_vap_liq
import coexist_asymm
import poisson_interface
import mgrf_asymm

from physical_param_asymm import *
start = timeit.default_timer()

# Argument parser to accept the input files                                                                                                                                                                        
parser = argparse.ArgumentParser(description='Code to calculate EDL structure using MGRF Theory with mean-field PB as an initial guess')
parser.add_argument('input_files', nargs='+', help='Paths to the input files for physical parameters')
args = parser.parse_args()

folder_path = os.path.dirname(args.input_files[0])
sys.path.insert(0, folder_path)

# Load the physical input configuration from the first file in the list                                                                                                                                            
module_name = os.path.splitext(os.path.basename(args.input_files[0]))[0]
input_physical = importlib.import_module(module_name)
variables = {name: value for name, value in input_physical.__dict__.items() if not name.startswith('__')}
(locals().update(variables))

concns_psi = [0.067545094841168,2.54891321086623,1.86014594829533]
n_bulk1, n_bulk2, psi2 = coexist_asymm.binodal(concns_psi,valency,rad_ions,vol_sol,epsilon_s)
print(n_bulk1,n_bulk2,psi2)
nconc_complete, psi_complete, domain,z = num_concn.nguess_asymm(n_bulk1,n_bulk2,psi2,valency,int_width1,int_width2,epsilon_s,N_grid)
#print(nconc_complete[0:5])
print("guess done")

# psi_complete = poisson_interface.poisson_interface(nconc_complete,valency,psi2,N_grid,domain,epsilon_s)
# print('Gauss Law_done')
# print(psi_complete[0:5])

## The EDL structure calculations start here
# psi_complete,nconc_complete,uself_complete, q_complete, z, res= mgrf_asymm.mgrf_asymm(psi_complete,nconc_complete,n_bulk1,n_bulk2,psi2,valency,rad_ions,vol_ions,vol_sol,domain,epsilon_s)
#
# print('MGRF_done')
# print(psi_complete[0:5])
# tension = energy_vap_liq.grandfe_mgrf_vap_liq(psi_complete,nconc_complete,uself_complete,n_bulk1,n_bulk2,psi2,valency,rad_ions,vol_ions,vol_sol,domain,epsilon_s)
#
# print('tension_star = ' + str(tension * 4 * pi * epsilon_s * pow(2 * rad_ions[0],3)/abs(valency[0] * valency[1])))
#
# stop = timeit.default_timer()
# print('Time: ', stop - start)
#
#
# file_dir = os.getcwd() + '/results' + str(abs(valency[0])) + '_' + str(abs(valency[1]))
# file_name = str(round(T_star, 5)) + '_' + str(round(float(int_width1), 2)) + '_' + str(round(float(int_width2), 2)) + '_' + str(round(rad_ions_d[0], 2))  + '_' + str(round(rad_ions_d[1], 2)) + '_' + str(round(rad_sol_d, 2))
#
# ### Create the output directory if it doesn't exist
#
# if not os.path.exists(file_dir):
#     os.mkdir(file_dir)
#
# # Writing everything in SI units
# with h5py.File(file_dir + '/mgrf_' + file_name + '.h5', 'w') as file:
#
#     # Storing scalar variables as attributes of the root group
#     file.attrs['ec_charge'] = ec
#     file.attrs['char_length'] = l_c
#     file.attrs['beta'] = beta
#     file.attrs['epsilon_s'] = epsilonr_s_d
#     file.attrs['int_width1'] = int_width1
#     file.attrs['int_width2'] = int_width2
#     file.attrs['domain'] = domain
#     file.attrs['domain_d'] = domain*l_c
#     file.attrs['psi2'] = psi2
#
#     # Storing numerical parameters as attributes of the root group
#     file.attrs['s_conv'] = s_conv
#     file.attrs['N_grid'] = len(uself_complete)
#     file.attrs['quads'] = quads
#     file.attrs['grandfe_quads'] = grandfe_quads
#     file.attrs['dealias'] = dealias
#     file.attrs['ncc_cutoff_mgrf'] = ncc_cutoff_mgrf
#     file.attrs['num_ratio'] = num_ratio
#     file.attrs['selfe_ratio'] = selfe_ratio
#     file.attrs['eta_ratio'] = eta_ratio
#     file.attrs['tolerance_mgrf'] = tolerance_mgrf
#     file.attrs['tolerance_pb'] = tolerance_pb
#     file.attrs['tolerance_num'] = tolerance_num
#     file.attrs['tolerance_greens'] = tolerance_greens
#     file.attrs['residual'] = res
#     file.attrs['c_max'] = c_max
#
#     # Storing parameter arrays
#     file.create_dataset('valency', data = valency)
#     file.create_dataset('radii', data = rad_ions_d)
#     file.create_dataset('volumes', data = np.concatenate((vol_ions_d,[vol_sol_d])))
#
#     # Store all spatial profiles  (SI units)
#     file.create_dataset('z_d', data = z*l_c)
#     file.create_dataset('psi_d', data = psi_complete*psi_c)
#     file.create_dataset('nconc_d', data = nconc_complete*nconc_c/N_A)
#     file.create_dataset('uself_d', data = uself_complete*(1/beta))
#     file.create_dataset('charge_d', data = q_complete*(nconc_c*ec))
#
#     # Store all spatial profiles (non-dimensional)
#     file.create_dataset('z', data = z)
#     file.create_dataset('psi', data = psi_complete)
#     file.create_dataset('nconc', data = nconc_complete)
#     file.create_dataset('uself', data = uself_complete)
#     file.create_dataset('charge',data = q_complete)
#     file.create_dataset('n_bulk1',data = n_bulk1)
#     file.create_dataset('n_bulk2',data = n_bulk2)
#
#     # Store free energy
#     file.attrs['tension'] = tension # nondimensional
#     file.attrs['tension_d'] = tension*(1/beta) # SI units
#     file.attrs['tension_star'] = tension*4*pi*epsilon_s*pow(2*rad_ions[0],3)/(valency[0]*valency[1])
#

plt.figure(2)

plt.plot(z, psi_complete/psi2,'m-')
plt.title('T$^*$ = ' + str(T_star) + ' M' + ' | radius = ' + str(round(rad_ions_d[0] , 2)) + ' $\AA$' + ' | domain = ' + str(domain))
plt.legend(loc='best')
plt.xlabel('z (position) in $\AA$',fontsize = 20)
plt.ylabel('$\psi$',fontsize = 20)
plt.grid()


plt.figure(3)

plt.plot(z, nconc_complete[:,0],'m-',label = '$C_{+}$')
plt.plot(z, nconc_complete[:,1],'b-',label = '$C_{-}$')
plt.title('T$^*$ = ' + str(T_star) + ' M' + ' | radius = ' + str(round(rad_ions_d[0] , 2)) + ' $\AA$' + ' | domain = ' + str(domain))
plt.legend(loc='best')
plt.xlabel('z (position) in $\AA$',fontsize = 20)
plt.ylabel('$c$',fontsize = 20)
plt.grid()
plt.show()

