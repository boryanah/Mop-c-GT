import numpy as np
from .params import cosmo_params
from .cosmo import rho_cz
from scipy.interpolate import Rbf

Msol_cgs = 1.989e33
rhocrit =  1.87847e-29 * cosmo_params['hh']**2
G_cgs = 6.67259e-8 #cm3/g/s2
kpc_cgs = 3.086e21
fb = cosmo_params['Omega_b']/cosmo_params['Omega_m']

def r200(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm.
    '''
    M_cgs = M*Msol_cgs
    om = cosmo_params['Omega_m']
    ol = cosmo_params['Omega_L']
    Ez2 = om * (1 + z)**3 + ol
    ans = (3 * M_cgs / (4 * np.pi * 200.*rhocrit*Ez2))**(1.0/3.0)
    return ans


def rho_gnfw1h(x, M, z, theta):
    '''generalized NFW profile describing the gas density [g/cm3],
    parameters have mass and redshift dependence, as in Battaglia 2016
    '''
    '''
    if isinstance(M, float) is True:
        r200c = r200(M,z)
        rvir = r200(M,z)/kpc_cgs/1e3 #Mpc
        xx = x/rvir
        #rho0 = 4e3 * (M/1e14)**0.29 * (1+z)**(-0.66)
        #xc = 0.5
        al = 0.88 * (M/1e14)**(-0.03) * (1+z)**0.19
        #bt = 3.83 * (M/1e14)**0.04 * (1+z)**(-0.025)
        gm = -0.2
        rho0,xc,bt = theta
        #rho0,al,bt = theta
        ans = 10**rho0 * (xx/xc)**gm / ((1 + (xx/xc)**al)**((bt+gm)/al)) # typo B.H.
        ans *= rho_cz(z) * fb
        return ans
    else:
    '''
    #logm = np.log10(M)
    #h,b_edges = np.histogram(logm, bins=9)
    #b_cen = np.array([(b_edges[i]+b_edges[i-1])*0.5 for i in range(1,len(b_edges))])
    #b_len = np.array([(b_edges[i]-b_edges[i-1]) for i in range(1,len(b_edges))])
    #integ = np.sum(10**b_len*h)
    #p = h/integ
    b_cen = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    rho = []
    for i in range(0, len(b_cen)):
        m = 10**b_cen[i]
        r200c = r200(m,z)
        rvir=r200(m,z)/kpc_cgs/1e3 #Mpc
        #xc = 0.5
        al = 0.88 * (m/1e14)**(-0.03) * (1+z)**0.19
        gm = -0.2
        #rho0,al,bt = theta
        rho0,xc,bt = theta
        rho.append(10**rho0 * (x/rvir/xc)**gm / ((1 + (x/rvir/xc)**al)**((bt+gm)/al))*rho_cz(z) * fb)    # B.H. typo
    rho = np.array(rho)
    rho_av = np.average(rho, weights=p, axis=0)
    return rho_av


def rho_gnfw2h(xx,theta2h):
    # TODO B.H. please change this, it makes me cry
    """
    rho_file = np.genfromtxt('/home/boryanah/repos/Mop-c-GT/data/twohalo_cmass_average.txt')
    x1 = rho_file[:,0]
    rho2h = rho_file[:,1]
    ans = np.interp(xx,x1,rho2h)
    """
    #data = np.load("data/rho_2h_gnfw_cmass.npz") # Mpc, Msun, g/cm^3
    data = np.load("data/rho_2h_gnfw_cmass_fit.npz") # Mpc, Msun, g/cm^3
    x1 = data['r'] # Mpc
    rho2h = data['rho_2h'] # g/cm^3
    ans = np.interp(xx,x1,rho2h)       
    return theta2h * ans

def rho_gnfw(xx,M,z,theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho_gnfw1h(xx,M,z,theta1h) + rho_gnfw2h(xx,theta2h)
    return ans


def Pth_gnfw1h(x,M,z,theta):
    '''generalized NFW profile describing the thermal pressure [cgs units],
    parameters have mass and redshift dependence, as in Battaglia et al. (2012)
    '''
    '''
    if isinstance(M, float) is True:
        r200c = r200(M,z)
        rvir=r200(M,z)/kpc_cgs/1e3 #Mpc
        x /= rvir
        M_cgs = M*Msol_cgs
        P200c = G_cgs * M_cgs* 200. * rho_cz(z) * fb /(2.*r200c)
        #P0, xc, bt = theta
        P0, al, bt = theta
        #P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
        #al = 1.0
        #bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
        xc = 0.497 * (M/1e14)**(-0.00865) * (1+z)**0.731
        gm = -0.3
        ans = P0 * (x/xc)**gm * (1+(x/xc)**al)**(-bt)
        ans *= P200c
        return ans
    else: 
    '''   
    #logm = np.log10(M)
    #h,b_edges = np.histogram(logm, bins=9)
    #b_cen = np.array([(b_edges[i]+b_edges[i-1])*0.5 for i in range(1,len(b_edges))])
    #b_len = np.array([(b_edges[i]-b_edges[i-1]) for i in range(1,len(b_edges))])
    #integ = np.sum(10**b_len*h)
    #p = h/integ
    b_cen = np.array([[11.31932504, 11.43785913, 11.57526319, 11.74539764, 11.97016907, 12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([2.94467222e-06, 2.94467222e-06, 2.94467222e-06, 1.47233611e-05, 3.38637305e-05, 4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    pth = []
    for i in range(0, len(b_cen)):
        m = 10**b_cen[i]
        r200c = r200(m,z)
        rvir=r200(m,z)/kpc_cgs/1e3 #Mpc
        M_cgs = m*Msol_cgs
        P200c = G_cgs * M_cgs* 200. * rho_cz(z) * fb /(2.*r200c)
        P0, al, bt = theta
        #al = 1.0
        xc = 0.497 * (m/1e14)**(-0.00865) * (1+z)**0.731
        gm = -0.3
        pth.append(P0 * (x/rvir/xc)**gm * (1+(x/rvir/xc)**al)**(-bt) * P200c)   
    pth = np.array(pth)
    pth_av = np.average(pth, weights=p, axis=0)
    return pth_av

def Pth_gnfw2h(xx,z,theta2h):
    pth_file = np.genfromtxt('/home/boryanah/repos/Mop-c-GT/data/twohalo_cmass_average.txt')
    x1 = pth_file[:,0]
    pth2h = pth_file[:,2]   
    ans = np.interp(xx,x1,pth2h)
    return theta2h * ans


def Pth_gnfw(xx,M,z,theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = Pth_gnfw1h(xx,M,z,theta1h) + Pth_gnfw2h(xx,z, theta2h)
    return ans
