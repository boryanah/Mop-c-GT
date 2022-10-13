import numpy as np

"""
cosmo_params = {

	'Omega_m':0.25,
	'hh':0.7,
	'Omega_L':0.75,
	'Omega_b':0.044,
	'rhoc_0':2.77525e2,
	'C_OVER_HUBBLE':2997.9
}
"""
cosmo_params = {
	'Omega_m': 0.3089,
	'hh': 0.6774,
	'Omega_L':0.6911,
	'Omega_b':0.0486,
        'ns': 0.9649,
        'sigma8': 0.8159
}

Msol_cgs = 1.989e33
G_cgs = 6.67259e-8
c_kms = 299792.458 
kpc_cgs = 3.086e21
rho_c_0 = 3. * (100.*1.e5/(1000.*kpc_cgs))**2 / (8. * np.pi * G_cgs) # cgs h^2 units, 1.87847e-29
cosmo_params['C_OVER_HUBBLE'] = c_kms/100. # h units
cosmo_params['rhoc_0'] = rho_c_0
