# vapor_liquid_mgrf

This is a Python package for solving the modified Gaussian renormalized fluctuation theory to get the interface between coexisting vapor and liquid phases in ionic fluids. The code is based on the equations derived in the work of Agrawal and Wang, [Phys. Rev. Lett. 2022, 129, 228001](https://doi.org/10.1103/PhysRevLett.129.228001) and [J. Chem. Theory Comput. 2022, 18, 6271–6280](https://doi.org/10.1021/acs.jctc.2c00607) and is written on top of open-source spectral methods based differential equation solver [Dedalus](https://github.com/DedalusProject/dedalus), developed by [Burns et al., Phys. Rev. Res. 2020, 2 (2), 023068](https://doi.org/10.1103/PhysRevResearch.2.023068). The iteration scheme for solving the non-linear equations in this code is partially adapted from the work of [Xu and Maggs J. Comp. Phys. 275 (2014): 310-322.](https://doi.org/10.1016/j.jcp.2014.07.004), the complete scheme and the method to solve the correlation functions will be soon published as a separate research article. The code solves for Gaussian correlation functions for symmetrical double layers in a parallel manner. Although the equations derived in the work of Agrawal and Wang account for spatially varying dielectric permittivities, the code in its current version is for systems with uniform dielectric permittivity. 

This code can be used to reproduce the data presented in Nikhil R. Agrawal and Rui Wang [Phys. Rev. Lett. 2022, 129, 228001](https://doi.org/10.1103/PhysRevLett.129.228001)

In the text that follows, the contents of various Python files are described.

## numerical_param.py

This is an input file to specify numerical parameters like the number of grid points, tolerance criteria, mixing ratios for non-linear solvers, ncc cutoffs for _Dedalus_, etc. Note that the equations being solved here are highly non-linear and hence some amount of tuning of these numerical parameters is required for efficient calculations.

## physical_param.py 

This is an input file to specify physical environment variables like temperature, ion valencies, born radii of ions, excluded volumes of ions and solvent, domain size, dielectric permittivity, etc. The temprature units are non-dimensional in the first part. In the second part of this file are some physical constants are characteristic variables used to non-dimensional environment variables. In the last part non dimensional of some other physical constants is done. All the calculations are done in these non-dimensional variables. 

## coexist_symm.py and coexist_asymm.py

The two files have one function each which calculate the bulk phase concentrations and their electrostatic potentials of the coexisting vapor and liquid phases. Without loss of generality vapor phase is labeled as phase 1 and liquid phase as phase 2. For symmeteric salts there is no galvani potential difference. For asymmteric salts potential of phase 1 is set to be zero, coexist_asymm two bulk phase concentrations and electrostatic potential of phase 2.

## pb_vap_liq.py

Solves the full mean-field Poisson-Boltzmann theory for electrical double layers next to a single charged plate. This is a non-linear boundary value problem whose initial guess can come from dh_vap_liq.py or another solution for mean-field PB. The solution of this can be used as an initial guess to solve for the modified Gaussian renormalized fluctuation theory in mgrf_vap_liq.py. 

## mgrf_symm.py and mgrf_asymm.py

calculates the electrostatic solution from modified Gaussian renormalized fluctuation theory for symmeteric and asymmeteric ionic fluids. For symmeteric salts, the electrostatic potential This is also a non-linear boundary value problem whose initial guess can come from pb_vap_liq.py or a solution for mgrf with another set of parameters. This function requires various properties like screening lengths, concentration profiles, self-energies, etc. Functions for these properties are described below.

## num_concn.py
Has three functions. nconc_mgrf calculates the coefficient in front of the exp(-z*psi) in the mgrf_vap_liq.py. It also outputs the coefficients that are needed to calculate the Jacobian for the Newton-Raphson iterative scheme. nconc_complete is the function to calculate the concentration profile for a given psi profile with n_initial as the initial guess. nconc_pb calculates number density profiles for mean-field PB. 

## selfe_vap_liq.py

This file includes functions to calculate the self-energy profiles for the interface based on the equations given in supplemental material of Agrawal and Wang, [Phys. Rev. Lett. 2022, 129, 228001](https://doi.org/10.1103/PhysRevLett.129.228001). The functions in this file use another file called greens_function_vap_liq.py which evaluates the fourier transform of the green's functions in the interface.

## selfe_bulk.py

This file includes functions to calculate the self-energy for the bulk solution based on the equations given in supplemental material of AAgrawal and Wang, [Phys. Rev. Lett. 2022, 129, 228001](https://doi.org/10.1103/PhysRevLett.129.228001). Note that although there is an analytical solution for self-energy in the bulk we calculate it numerically to cancel out the numerical errors between self-energy in interface and the bulk. The functions in this file use another file called greens_function_bulk.py which evaluates the fourier transform of the green's functions in the bulk.

## greens_function_vap_liq.py and greens_function_bulk.py

File to calculate Fourier transforms of G and Go in the interface and bulk respectively.

## calculate.py

This file contains functions to evaluate properties like screening length, ionic strength, incompressibility fields, and charge density profiles. There is also a function called interpolator to interpolate electrostatic potential and ion density profiles to increase or decrease grid points. A function to calculate the residual of Gauss law is also given.

## energy_vap_liq.py

Functions to calculate the grand free energy of the interface and bulk for the modified Gaussian renormalized fluctuation theory.

## simulator_symm.py and simulator_asymm.py

This code saves the solution of the modified Gaussian renormalized fluctuation theory for symmeteric and asymmteric salts respectively in a .h5 file. This file uses the solution of nguess_symm/nguess_asymm in num_concn.py as the initial guess to solve for mgrf_symm.py/mgrf_asymm. The input variables can be set separately using the file physical_param_symm.py/physical_param_asymm.py and numerical_param.py

## simulator_symm_mgrf.py and simulator_asymm_mgrf.py

This code saves the solution of the modified Gaussian renormalized fluctuation theory in a .h5 file. This file uses a saved solution of mgrf_vap_liq.py as the initial guess. The physical variables for this saved solution and the final parameters for which we want the interface structure can be set separately using the file physical_param_symm.py. The variables deciding which saved solution to choose as initial guess end with "_in_d", for ex: int_width_in or T_star_in. 2*int_width*(debye huckle length in phase 1) is equal to size of the domain.

## packages.py

This Python file contains the import statements for all the Python libraries that are needed for this package. We suggest that you create a separate conda environment where all these libraries are installed.

## Running the code

The code can be run using any of the following commands based on your needs: 

python simulator_symm.py(or simulator_symm_mgrf.py) physical_param_symm.py

python simulator_asymm.py(or simulator_asymm_mgrf.py) physical_param_asymm.py

Note that numerical_param.py has been directly imported into simulator files as the numerical parameters are seldom changed. However one can easily parse them through the command line by changing the first section of simulator files.

## Contact:
This code was developed by Nikhil Agrawal in the lab of Prof. Rui Wang, Pitzer Center for Theoretical Chemistry, University of California, Berkeley, USA. If you need any help feel free to write to Nikhil at nikhilagrawal0165@gmail.com.  

