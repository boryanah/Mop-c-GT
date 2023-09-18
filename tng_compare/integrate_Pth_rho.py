"""
need to do rho cause doing only pth for now n_e = 0.5*(X_H+1)/m_amu rho_gas
"""
import sys
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import astropy.units as u

"""
python integrate_Pth_rho.py 0.3 r200m; python integrate_Pth_rho.py 0.5 r200m; python integrate_Pth_rho.py 0.3 r200c; python integrate_Pth_rho.py 0.5 r200c; python integrate_Pth_rho.py 0.3 r200t; python integrate_Pth_rho.py 0.5 r200t
"""

# constants cgs
gamma = 5/3.
k_B = 1.3807e-16 # erg/T
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10
X_H = 0.76
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cm^2/K
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params
h = 67.74/100.
Omega_m = 0.3089
T_CMB = 2.7255 # K
Lbox_hkpc = 205000. # ckpc/h
unit_mass = 1.e10*(solar_mass/h) # g
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3

# choices
want_rho = True
z = float(sys.argv[1]) # 0.3, 0.5
integrate_to = sys.argv[2] #"r200m"; "r200c"; "r200t"
l_bound = 0.5/(1+z)*Lbox_hkpc/h # half the box size in physical kpc units

# load the data
if want_rho:
    data = np.load(f"data/rhoGNFW_z{z:.1}.npz")
    rho_arr = data['rho_arr']
else:
    data = np.load(f"data/PthGNFW_z{z:.1}.npz")
    Pth_arr = data['Pth_arr']
rbinc_ratio = data['rbinc_ratio']
r200m_kpcs = data['r200m_kpcs']
mbinc_Msuns = data['mbinc_Msuns']
r200c_kpcs = data['r200c_kpcs']
r200t_kpcs = data['r200t_kpcs']

# initiate arrays
tSZ_one_Mpc = np.zeros(len(mbinc_Msuns))
tSZ_two_Mpc = np.zeros(len(mbinc_Msuns))
r_max_kpcs = np.zeros(len(mbinc_Msuns))
for i in range(len(mbinc_Msuns)):
        
    # read off
    if integrate_to == "r200m":
        r_max_kpcs[i] = r200m_kpcs[i] # kpc
    elif integrate_to == "r200c":
        r_max_kpcs[i] = r200c_kpcs[i] # kpc
    elif integrate_to == "r200t":
        r_max_kpcs[i] = r200t_kpcs[i] # kpc
    r_rat_kpcs = r200c_kpcs[i] # kpc # note that the ratio is with respect to r200c
    
    # could choose different bound
    #l_bound = 10*r_max_kpcs[i]

    rbinc = rbinc_ratio * r_rat_kpcs # kpc
    lbinc = np.linspace(0., 10000., 1000) # kpc
    if not want_rho:
        P_one = Pth_arr[:, 0, i]
        P_one[rbinc > r_max_kpcs[i]] = 0. # get rid of one-halo signal beyond r_max
        P_two = Pth_arr[:, 1, i]
        P_one_f = interp1d(rbinc, P_one, bounds_error=False, fill_value=0.)# (P_one[0], 0) same
        P_two_f = interp1d(rbinc, P_two, bounds_error=False, fill_value=0.)

        # one-halo term (sphere only differs from cylinder by 3% due to geometry)
        # sphere version (single integral)
        # trapz and quad are almost equivalent for sphere, but quad is classier
        #tSZ_one_Mpc[i] = np.trapz(P_one*sigma_T/(m_e*c**2)*4.*np.pi*rbinc**2*kpc_to_cm/(1000.)**2., rbinc) # Mpc
        intgrnd_one = lambda r: P_one_f(r)*sigma_T/(m_e*c**2)*4.*np.pi*r**2*kpc_to_cm/(1000.)**2.*(2.+2.*X_H)/(3.+5.*X_H)
        tSZ_one_Mpc[i] = quad(intgrnd_one, 0, r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        # cylinder version (double integral)
        """
        tSZ_R = np.zeros(len(rbinc))
        for j in range(len(rbinc)):
            intgrnd_one = lambda l: 2.*P_one_f(np.sqrt(l**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2.
            tSZ_R[j] = quad(intgrnd_one, 0., 10*r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        tSZ_one_Mpc[i] = np.trapz(tSZ_R, rbinc)
        """

        # two-halo term (sphere doesn't make sense)
        # sphere version (single integral)
        #tSZ_two_Mpc[i] = np.trapz(P_two*sigma_T/(m_e*c**2)*4.*np.pi*rbinc**2*kpc_to_cm/(1000.)**2., rbinc) # Mpc
        # cylinder version (double integral); trapz introduces more error
        tSZ_R = np.zeros(len(rbinc))
        for j in range(len(rbinc)):
            intgrnd_two = lambda l: 2.*P_two_f(np.sqrt(l**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2.*(2.+2.*X_H)/(3.+5.*X_H)
            tSZ_R[j] = quad(intgrnd_two, 0., l_bound, epsabs=0.0, epsrel=1.e-4, limit=10000)[0] # 3 better than 10
            #tSZ_R[j] = 2.*np.trapz(P_two_f(np.sqrt(lbinc**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2., lbinc) # Mpc
        tSZ_R_f = interp1d(rbinc, tSZ_R, bounds_error=False, fill_value=0.)
        intgrnd_two = lambda r: tSZ_R_f(r)
        tSZ_two_Mpc[i] = quad(intgrnd_two, 0., r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        #tSZ_two_Mpc[i] = np.trapz(tSZ_R, rbinc)
        print("---------------------------")
    else:
        rho_one = rho_arr[:, 0, i]
        # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #rho_one[rbinc > r_max_kpcs[i]] = 0. # get rid of one-halo signal beyond r_max
        rho_two = rho_arr[:, 1, i]
        rho_one_f = interp1d(rbinc, rho_one, bounds_error=False, fill_value=0.)# (rho_one[0], 0) same
        rho_two_f = interp1d(rbinc, rho_two, bounds_error=False, fill_value=0.) # g/cm^3

        # TESTING!!!!
        r_max_kpcs[i] *= 5
        
        # one-halo term (sphere only differs from cylinder by 3% due to geometry)
        # sphere version (single integral)
        # trapz and quad are almost equivalent for sphere, but quad is classier
        #tSZ_one_Mpc[i] = np.trapz(rho_one*sigma_T/(m_e*c**2)*4.*np.pi*rbinc**2*kpc_to_cm/(1000.)**2., rbinc) # Mpc
        intgrnd_one = lambda r: rho_one_f(r)*4.*np.pi*r**2*kpc_to_cm**3/solar_mass
        Mfb = quad(intgrnd_one, 0, r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        # cylinder version (double integral)
        fb = 0.044/0.25
        print(f"fb m200c {fb*mbinc_Msuns[i]:.2e}")
        print(f"integral of gas {Mfb:.2e}")
        quit()
        """
        tSZ_R = np.zeros(len(rbinc))
        for j in range(len(rbinc)):
            intgrnd_one = lambda l: 2.*rho_one_f(np.sqrt(l**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2.
            tSZ_R[j] = quad(intgrnd_one, 0., 10*r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        tSZ_one_Mpc[i] = np.trapz(tSZ_R, rbinc)
        """

        # two-halo term (sphere doesn't make sense)
        # sphere version (single integral)
        #tSZ_two_Mpc[i] = np.trapz(rho_two*sigma_T/(m_e*c**2)*4.*np.pi*rbinc**2*kpc_to_cm/(1000.)**2., rbinc) # Mpc
        # cylinder version (double integral); trapz introduces more error
        tSZ_R = np.zeros(len(rbinc))
        for j in range(len(rbinc)):
            intgrnd_two = lambda l: 2.*rho_two_f(np.sqrt(l**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2.*(2.+2.*X_H)/(3.+5.*X_H)
            tSZ_R[j] = quad(intgrnd_two, 0., l_bound, epsabs=0.0, epsrel=1.e-4, limit=10000)[0] # 3 better than 10
            #tSZ_R[j] = 2.*np.trapz(rho_two_f(np.sqrt(lbinc**2. + rbinc[j]**2))*sigma_T/(m_e*c**2)*2.*np.pi*rbinc[j]*kpc_to_cm/(1000.)**2., lbinc) # Mpc
        tSZ_R_f = interp1d(rbinc, tSZ_R, bounds_error=False, fill_value=0.)
        intgrnd_two = lambda r: tSZ_R_f(r)
        tSZ_two_Mpc[i] = quad(intgrnd_two, 0., r_max_kpcs[i], epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
        #tSZ_two_Mpc[i] = np.trapz(tSZ_R, rbinc)
        print("---------------------------")    
print(tSZ_one_Mpc)
print(tSZ_two_Mpc)
# record tSZ signal
np.savez(f"data/tSZ_{integrate_to}_z{z:.1}.npz", mbinc_Msun=mbinc_Msuns, r_max_kpc=r_max_kpcs, r200t_kpc=r200t_kpcs, r200m_kpc=r200m_kpcs, r200c_kpc=r200c_kpcs, tSZ_one_Mpc=tSZ_one_Mpc, tSZ_two_Mpc=tSZ_two_Mpc)
