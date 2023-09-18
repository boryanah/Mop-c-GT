"""
I think this is now completely independent of the rest of mop-c-gt
"""
import numpy as np
from scipy.integrate import quad
from .params import cosmo_params
from hmf import MassFunction, transfer
from colossus.lss import bias, peaks, mass_function
from colossus.cosmology import cosmology

'''set cosmology'''
#params = {'flat': True, 'H0': 70., 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8159, 'ns': 0.97}

'''parameters used in IllustrisTNG https://arxiv.org/pdf/1703.02970.pdf'''
params = {'flat': True, 'H0': cosmo_params['hh']*100., 'Om0': cosmo_params['Omega_m'], 'Ob0': cosmo_params['Omega_b'], 'sigma8': cosmo_params['sigma8'], 'ns': cosmo_params['ns']}

cosmo = cosmology.setCosmology('myCosmo',params)

# constants
G_cgs = 6.67259e-8 #cm3/g/s2
Msol_cgs = 1.989e33 #g
kpc_cgs = 3.086e21 #cm

# cosmo params
hh = cosmo_params['hh']
fb = params['Ob0']/params['Om0']
Om0 = params['Om0']

#################################################################################
# Computing the 2-halo component of density and pressure profiles in cgs units #
#################################################################################

def rho_cz(z):
    '''critical density in cgs
    '''
    Ez2 = cosmo_params['Omega_m']*(1+z)**3. + (1-cosmo_params['Omega_m'])
    return cosmo_params['rhoc_0'] * cosmo_params['hh']**2. * Ez2

def rho_mz(z):
    '''mean density in cgs
    '''
    Mz2 = cosmo_params['Omega_m']*(1+z)**3.
    return cosmo_params['rhoc_0'] * cosmo_params['hh']**2. * Mz2

def r200crit(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm. 
    '''
    M_cgs = M*Msol_cgs
    ans = (3 * M_cgs / (4 * np.pi * 200.*rho_cz(z)))**(1.0/3.0)
    return ans

def r_virial(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm. 
    '''
    M_cgs = M*Msol_cgs
    x = cosmo_params['Omega_m']*(1.+z)**3/(cosmo_params['Omega_m']*(1.+z)**3 + (1.-cosmo_params['Omega_m'])) - 1. #Omega_m(z) - 1
    Delta_c = 18.*(np.pi**2) + 82.*x - 39.*x*x # virial overdensity wrt critical density
    #Delta_m = Delta_c/Omega_m(z) # wrt mean density
    ans = (3 * M_cgs / (4 * np.pi * Delta_c*rho_cz(z)))**(1.0/3.0)
    return ans

def r200mean(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm. 
    '''
    M_cgs = M*Msol_cgs
    ans = (3 * M_cgs / (4 * np.pi * 200.*rho_mz(z)))**(1.0/3.0)
    return ans

def r200c_kpc(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in kpc. # B.H. added
    '''
    ans = r200crit(M, z) # cm
    ans /=  kpc_cgs
    return ans

def r200m_kpc(M, z):
    '''radius of a sphere with density 200 times the matter density of the universe.
    Input mass in solar masses. Output radius in kpc. # B.H. added
    '''
    ans = r200mean(M, z) # cm
    ans /=  kpc_cgs
    return ans

def r_vir_kpc(M, z):
    '''radius of a sphere with density 200 times the matter density of the universe.
    Input mass in solar masses. Output radius in kpc. # B.H. added
    '''
    ans = r_virial(M, z) # cm
    ans /=  kpc_cgs
    return ans

def r200t_kpc(M, z):
    """ TNG definition """
    return r_vir_kpc(M, z)

#From Battaglia 2016, Appendix A
def rho_gnfw(x,m,z):
    rho200c = rho_cz(z)*fb
    rho0 = 4e3 * (m/1e14)**0.29 * (1+z)**(-0.66)
    al = 0.88 * (m/1e14)**(-0.03) * (1+z)**0.19
    bt = 3.83 * (m/1e14)**0.04 * (1+z)**(-0.025)
    xc = 0.5
    gm = -0.2
    #ans = rho0 * (x/xc)**gm * (1+(x/xc)**al)**(-(bt-gm)/al)
    ans = rho0 * (x/xc)**gm * (1+(x/xc)**al)**(-(bt+gm)/al)
    ans *= rho200c
    return ans

def rhoFourier(k, m, z):
    ans = []
    for i in range(len(m)):
        r200c = r200crit(m[i],z)/kpc_cgs/1e3
        rvir = r_virial(m[i],z)/kpc_cgs/1e3
        integrand = lambda r: 4.*np.pi*r**2*rho_gnfw(r/r200c,m[i],z) * np.sin(k * r)/(k*r)
        res = quad(integrand, 0., 3*rvir, epsabs=0.0, epsrel=1.e-4, limit=10000)[0] # 10 times r200c originally B.H.
        ans.append(res)
    ans = np.array(ans)
    return ans

def hmf(m, z, mdef='200c'):
    '''Shet, Mo &  Tormen 2001'''
    """
    # expects h units (returns natively h^4 Msun^-1 Mpc^3
    Mmin = np.log10(np.min(m) * cosmo_params['hh'])
    Mmax = np.log10(np.max(m) * cosmo_params['hh'])
    return MassFunction(z=z, Mmin=Mmin, Mmax=Mmax, dlog10m=(Mmax-Mmin)/49.5, hmf_model="Behroozi").dndm * cosmo_params['hh']**4. #"SMT").dndm # "Tinker08"
    """
    dndlnm = mass_function.massFunction(m * cosmo_params['hh'], z, mdef=mdef, model='tinker08', q_in='M', q_out='dndlnM')
    dndm = dndlnm / m
    dndm *= cosmo_params['hh']**3.
    return  dndm
    
    
def b(m, z, mdef='200c'):
    '''Shet, Mo &  Tormen 2001'''
    """
    nu = peaks.peakHeight(m, z)
    delta_c = peaks.collapseOverdensity(corrections=True, z=z)
    aa, bb, cc = 0.707, 0.5, 0.6
    return 1.+ 1./(np.sqrt(aa)*delta_c) * (np.sqrt(aa)*aa*nu**2 + np.sqrt(aa)*bb*(aa*nu**2)**(1-cc) - ((aa*nu**2)**cc/(aa*nu**2)**cc+bb*(1-cc)*(1-cc/2)))
    """
    # expects h units
    b = bias.haloBias(m*cosmo_params['hh'], model='tinker10', z=z, mdef=mdef)
    return b

def Plin(k,z):
    # expects h units
    lnk_min = np.log(np.min(k) / cosmo_params['hh'])
    lnk_max = np.log(np.max(k) / cosmo_params['hh'])
    dlnk = (lnk_max-lnk_min)/(49.5)
    '''Eisenstein & Hu (1998)'''
    p = transfer.Transfer(sigma_8=params['sigma8'], n=params['ns'], z=z, lnk_min=lnk_min, lnk_max=lnk_max, dlnk=dlnk, transfer_model="CAMB")#"EH")
    #power = p.power/cosmo_params['hh']**3 # Mpc^3
    power = p.nonlinear_power/cosmo_params['hh']**3 # Mpc^3 # overshoots
    """
    power = cosmo.matterPowerSpectrum(k / cosmo_params['hh'], z = z)
    power /= cosmo_params['hh']**3 # Mpc^3
    """
    return power

def rho_2h(r,m,z):

    #first compute P_2h (power spectrum)
    m_array = np.logspace(np.log10(1.e10), np.log10(1.e15), 50, 10.) # Msun
    k_array = np.logspace(np.log10(1.e-3), np.log10(1.e3), 50, 10.) # 1/Mpc
    hmf_array = np.array([hmf(m_array,z)]*len(k_array)).reshape(len(k_array),len(hmf(m_array,z)))
    bias_array = np.array([b(m_array,z)]*len(k_array)).reshape(len(k_array),len(b(m_array,z)))

    arr = []
    for i in range(len(k_array)):
        arr.append(np.trapz(hmf_array[i,:]*bias_array[i,:]*rhoFourier(k_array[i],m_array,z),m_array))
    arr = np.array(arr)
    P2h = np.array(arr * b(m,z)  * Plin(k_array,z))

    #then Fourier transform to get rho_2h
    rcorr = 50. #Mpc/h # B.H. need to do something more elegant
    integrand = lambda k: 1./(2*np.pi**2.) * k**2 * np.sin(k*r)/(k*r) * np.interp(k, k_array, P2h) if k>1./rcorr else 0.0
    res = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.e-2, limit=1000)[0]
    return res



#From Battaglia 2012, AGN Feedback Delta=200
def Pth_gnfw(x,m,z):
    P200c = G_cgs * m*Msol_cgs * 200. * rho_cz(z) * fb /(2.*r200crit(m,z))
    P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
    al = 1.0
    bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
    xc = 0.497 * (m/1e14)**(-0.00865) * (1+z)**0.731
    gm = -0.3
    ans = P0 * (x/xc)**gm * (1+(x/xc)**al)**(-bt)
    ans *= P200c
    return ans

def PthFourier(k, m, z):
    ans = []
    for i in range(len(m)):
        r200c = r200crit(m[i],z)/kpc_cgs/1e3
        rvir = r_virial(m[i],z)/kpc_cgs/1e3
        integrand = lambda r: 4.*np.pi*r**2*Pth_gnfw(r/r200c,m[i],z) * np.sin(k * r)/(k*r)
        res = quad(integrand, 0., 3*rvir, epsabs=0.0, epsrel=1.e-4, limit=10000)[0] # 10 times r200c originally # B.H.
        ans.append(res)
    ans = np.array(ans)
    return ans

def Pth_2h(r, m, z):
    #first compute P_2h (power spectrum)
    m_array = np.logspace(np.log10(1.e10), np.log10(1.e15), 50, 10.) # Msun
    k_array = np.logspace(np.log10(1.e-3), np.log10(1.e3), 50, 10.) # 1/Mpc
    hmf_array = np.array([hmf(m_array,z)]*len(k_array)).reshape(len(k_array),len(hmf(m_array,z)))
    bias_array = np.array([b(m_array,z)]*len(k_array)).reshape(len(k_array),len(b(m_array,z)))

    arr = []
    for i in range(len(k_array)):
        arr.append(np.trapz(hmf_array[i,:]*bias_array[i,:]*PthFourier(k_array[i],m_array,z),m_array))
    arr = np.array(arr)
    P2h = np.array(arr * b(m,z)  * Plin(k_array,z))

    #then Fourier transform to get Pth_2h 
    #rcorr = 50. #Mpc/h # B.H. should only affect kSZ
    integrand = lambda k: 1./(2*np.pi**2.) * k**2 * np.sin(k*r)/(k*r) * np.interp(k, k_array, P2h) # if k>1./rcorr else 0.0
    res = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.e-2, limit=1000)[0]
    return res

