import numpy as np
from scipy.special import spence
from scipy.optimize import fmin,differential_evolution,minimize
from scipy.optimize import newton
from .params import cosmo_params

delx = 0.01
Msol_cgs = 1.989e33
Gravity = 6.67259e-8
rhocrit =  1.87847e-29 * cosmo_params['hh']**2
C_CGS = 2.99792e+10
kpc_cgs = 3.086e21

fb = cosmo_params['Omega_b']/cosmo_params['Omega_m']


def nfw(x):
    '''shape of a NFW profile (NFW 1997, ApJ,490, 493)
    '''
    ans = 1./(x*(1 + x)**2)
    return ans

def gx(x):
    ans = (np.log(1. + x) - x/(1. + x))
    return ans

def gc(c):
    ans = 1./(np.log(1. + c) - c/(1. + c))
    return ans

def Hc(c):
    ans = (-1.*np.log(1 + c)/(1. + c) + c*(1. + 0.5*c)/((1. + c)**2))/gx(c)
    return ans

def Sc(c):
    ans = (0.5*np.pi**2 - np.log(c)/2. - 0.5/c - 0.5/(1 + c)**2 - 3/(1 + c) +
           np.log(1 + c)*(0.5+0.5/c**2-2/c-1/(1+c)) +
           1.5*(np.log(1 + c))**2 + 3.*spence(c+1))

    return ans

def del_s(c):
    ans = Sc(c) / (Sc(c) + (1./c**3)*Hc(c)*gx(c))
    return ans

def K_c(c): 
    ans = 1./3.* Hc(c)/(1.-del_s(c))
    return ans


def sig_dm2(x,c): 
    '''EQ 14 Lokas & Mamon 2001
    '''
    ans = 0.5*x*c*gc(c)*(1 + x)**2 *(np.pi**2 - np.log(x) - (1./x)
                                     - (1./(1. + x)**2) - (6./(1. + x))
                                     + np.log(1. + x)*(1. + (1./x**2) - 4./x - 2/(1 + x))
                                    + 3.*(np.log(1. + x))**2 + 6.*spence(x+1))
    return ans


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

def con(M, z):
    '''
    concentration parameter from Duffy et al. (2008)
    input mass in solar masses
    '''
    ans = 5.71 / (1 + z)**(0.47) * (M / 2e12)**(-0.084)
    return ans

def rho_dm_x(x,M,z,trunc):
    #NFW profile describing the dark matter density [g/cm3]
    #
    c = con(M,z)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs
    ans = M_cgs*(c/rvir)**3 / (4.*np.pi*gx(c)) * nfw(x)
    if isinstance(x, float):
        if x > trunc*c: ans = 0.
    else:
        ans[x > trunc*c] = 0.
    return ans

def rho_dm(r,M,z,trunc):
    #input: r in Mpc M in Msun, trunc in rvir units
    c = con(M,z)
    rvir = r200(M,z) #cgs
    rs = rvir / c #cgs
    M_cgs = M*Msol_cgs #cgs
    rhoS = M_cgs / (4.*np.pi*rs**3)
    rhoS /= np.log(1.+c) - c/(1.+c)
    x = r*kpc_cgs*1e3 / rs
    ans =  rhoS / (x*(1.+x)**2)
    if isinstance(x, float):
        if x > trunc*c: ans = 0.
    else:
        ans[x > trunc*c] = 0.
    return ans

def rho_dm_og(r,M,z,trunc):
    #input: r in Mpc M in Msun, trunc in rvir units
    #mhalo = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T        #CMASS
    #pm = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])   #CMASS
    mhalo = np.array([[13.06930356, 13.32095674, 13.58460136, 13.85897045]]).T
    pm = np.array([0.03441325, 0.14584457, 0.41102333, 0.15162467])
    rhow = []
    for im in range(0,len(mhalo)):
        m = 10**mhalo[im]
        ww = pm[im]
        c = con(m,z)
        rvir = r200(M,z) #cgs
        rs = rvir / c #cgs
        M_cgs = m*Msol_cgs #cgs
        rhoS = M_cgs / (4.*np.pi*rs**3)
        rhoS /= np.log(1.+c) - c/(1.+c)
        x = r*kpc_cgs*1e3 / rs
        ans =  rhoS / (x*(1.+x)**2)
        if isinstance(x, float):
            if x > trunc*c: ans = 0.
        else:
            ans[x > trunc*c] = 0.
        rhow.append(ans)
    rho = np.average(rhow, weights=pm, axis = 0)
    return rho


def jx(x,c):
    ans = 1. - np.log(1. + x)/x
    ind = np.where(x > c) #[0]
    if (len(ind) > 0):
        ans[ind] = 1. -1./(1. + c) - (np.log(1. + c) - c/(1.+c))/x[ind]
    return ans

def jx_f(x,c):
    if (x <= c):
        ans = 1. - np.log(1. + x)/x
    else:
        ans = 1. -1./(1. + c) - (np.log(1. + c) - c/(1.+c))/x
    return ans

def fx (x,c):
    ans = np.log(1. + x)/x - 1./(1. + c)
    ind = np.where(x > c)[0]
    if (len(ind) > 0):
        ans = (np.log(1. + c)/c - 1./(1. + c))*c/x
    return ans

def fstar_func(M):
    '''Giodini 2009, modified by 0.5
    '''
    ans = 2.5e-2 * (M / (7e13))**(-0.37) 
    return ans

def xs_min_func(x,M,z):
    c = con(M,z)
    fstar = fstar_func(M)
    ans = gx(c)*fstar/(1. + fstar) - gx(x)
    return ans

def xs_func(M,z):
    x0 = 1.0
    xs = newton(xs_min_func, x0, args=(M,z,))
    return xs

def Ks(x_s,M,z):
    c = con(M,z)
    xx = np.arange(delx/2.,x_s,delx)
    ans = 1./gx(c)*(np.sum(Sc(xx)*xx**2) - 2./3.*np.sum(fx(xx,c)*xx/(1. + xx)**2) )*delx
    return ans

def n_exp(gamma):
    '''exponent of the polytopic e.o.s.
    '''
    ans = 1. / (gamma - 1)
    return ans

def theta_func(x,M,z,theta,theta2):
    '''polytropic variable
    '''
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(M,z)
    nn = n_exp(gamma)
    #print(jx(x,c).shape)
    ans = (1. - beta*jx(x,c)/(1. + nn))
    return ans

def theta_func_f(x,M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(M,z)
    nn = n_exp(gamma)
    ans = (1. - beta*jx_f(x,c)/(1. + nn))
    return ans

def rho_use(x,M,z,theta, theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x,M,z,theta,theta2))**nn
    return ans

def rho1h_one_mass(x,M,z,theta):
    gamma,alpha,Ef = theta
    theta2 = find_params_M(M,z,theta) 
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(M,z)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs
    beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = rho_0*(theta_func(x/(rvir/kpc_cgs/1e3)*c,M,z,theta,theta2_use))**nn
    return ans

def rho1h(x,M,z,theta):
    gamma,alpha,Ef = theta
    nn = n_exp(gamma)
    b_cen = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    rh = []
    for i in range(0, len(b_cen)):
        m = 10**b_cen[i]
        thta2 = find_params_M(m,z,theta)
        P_0, rho_0, x_f = thta2
        c = con(m,z)
        rvir = r200(m,z)
        M_cgs = m*Msol_cgs
        beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
        theta2_use = beta, x_f
        rh.append(rho_0*(theta_func(x/(rvir/kpc_cgs/1e3)*c,M,z,theta,theta2_use))**nn)
    rh = np.array(rh)
    rh_av = np.average(rh, weights=p, axis=0)
    return rh_av

def rho2h(xx,theta2h):
    rho_file = np.genfromtxt('/Users/samodeo/Desktop/Mop-c-GT-master/data/twohalo_cmass_average.txt')
    x1 = rho_file[:,0]
    rho2h = rho_file[:,1]
    ans = np.interp(xx,x1,rho2h)
    return theta2h * ans

def rho(x,M,z,theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho1h_one_mass(x,M,z,theta1h) + rho2h(x,theta2h)
    return ans

def rho_outtest(x,M,z,theta, theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(M,z)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs
    beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
    theta2_use = beta, x_f
    #print("inside rhoout", theta2_use)
    ans = rho_0*(theta_func(x,M,z,theta,theta2_use))**nn
    return ans

def Pnth_th(x,M,z,theta):
    gamma,alpha,Ef = theta
    c = con(M,z)
    ans = 1. - alpha*(x/c)**0.8
    return ans

def Pth1h_one_mass(x,M,z,theta):
    gamma,alpha,Ef = theta
    theta2 = find_params_M(M,z,theta) 
    P_0, rho_0, x_f = theta2
    c = con(M,z)
    rvir = r200(M,z)
    nn = n_exp(gamma)
    M_cgs = M*Msol_cgs
    beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = P_0*(theta_func(x,M,z,theta,theta2_use))**(nn+1.) * Pnth_th(x,M,z,theta)
    return ans

def Pth1h(x,M,z,theta):
    gamma,alpha,Ef = theta
    nn = n_exp(gamma)
    b_cen = np.array([[11.31932504, 11.43785913, 11.57526319, 11.74539764, 11.97016907, 12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([2.94467222e-06, 2.94467222e-06, 2.94467222e-06, 1.47233611e-05, 3.38637305e-05, 4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    pth = []
    for i in range(0, len(b_cen)):
        m = 10**b_cen[i]
        thta2 = find_params_M(m,z,theta)
        P_0, rho_0, x_f = thta2
        c = con(m,z)
        rvir = r200(m,z)
        M_cgs = m*Msol_cgs
        beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
        theta2_use = beta, x_f
        pth.append(P_0*(theta_func(x/(rvir/kpc_cgs/1e3)*c,m,z,theta,theta2_use))**(nn+1.) * Pnth_th(x/(rvir/kpc_cgs/1e3)*c,m,z,theta))   
    pth = np.array(pth)
    pth_av = np.average(pth, weights=p, axis=0)
    return pth_av

def Pth2h(xx,theta2h):
    pth_file = np.genfromtxt('/Users/samodeo/Desktop/Mop-c-GT-master/data/twohalo_cmass_average.txt')
    x1 = pth_file[:,0]
    pth2h = pth_file[:,2]   
    ans = np.interp(xx,x1,pth2h)
    return theta2h * ans


def Pth(x,M,z,theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = Pth1h(x,M,z,theta1h) + Pth2h(x,theta2h)
    return ans

def Pth_use(x,M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x,M,z,theta,theta2))**(nn+1.) * Pnth_th(x,M,z,theta)
    return ans

def Ptot(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    rvir = r200(M,z)
    c = con(M,z)
    M_cgs = M*Msol_cgs
    beta = rho_0/P_0 * Gravity*M_cgs/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = P_0*(theta_func_f(x_f,M,z,theta,theta2_use))**(nn+1.)
    return ans

def Ptot_use(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func_f(x_f,M,z,theta,theta2))**(nn+1.)
    return ans

def Pnth(x,M,z,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    c = con(M,z)
    nn = n_exp(gamma)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = alpha*(x/c)**0.8 * P_0*(theta_func(x,M,z,theta,theta2_use))**(nn+1.)
    return ans

def Pnth_use(x,M,z,theta,theta2):
    gamma,alpha,Ef = theta
    c = con(M,z)
    nn = n_exp(gamma)
    ans = alpha*(x/c)**0.8 * (theta_func(x,M,z,theta,theta2))**(nn+1.)
    return ans

def I2_int(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    c = con(M,z)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(fx(xx,c)*rho_use(xx,M,z,theta,theta2)*xx**2)*delx
    return ans

def I3_int(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(Pth_use(xx,M,z,theta,theta2) *xx**2)*delx
    return ans

def I4_int(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(Pnth_use(xx,M,z,theta,theta2)*xx**2)*delx
    return ans

def L_int(M,z,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(rho_use(xx,M,z,theta,theta2)*xx**2)*delx
    return ans

def rho_0_func(theta0,theta2):
    M,z,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    c = con(M,z)
    rvir = r200(M,z)
    fstar = fstar_func(M)
    M_cgs = M*Msol_cgs
    ans = M_cgs*(fb-fstar) / (4.*np.pi * L_int(M,z,theta,theta2)*(rvir/c)**3)
    return ans

def P_0_func(theta0,theta2,rho_0):
    M,z,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    c = con(M,z)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs
    ans = rho_0/beta * Gravity*M_cgs/rvir*c/gx(c)
    return ans

def findroots2(theta2,theta0):
    M,z,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(M,z)
    rvir = r200(M,z)
    x_s = xs_func(M,z)
    fstar = fstar_func(M)
    M_cgs = M*Msol_cgs

    E_inj = Ef * gx(c) * rvir * fstar / (Gravity*M_cgs*c) * C_CGS**2

    Eq1 = (3./2.*(1. + fstar) * (K_c(c)*(3.-4.*del_s(c)) + Ks(x_s,M,z))  - E_inj + 1./3.* (1.+fstar) *Sc(c) / gx(c) * (x_f**3 - c**3)
           - I2_int(M,z,theta,theta2)/L_int(M,z,theta,theta2)
           + 3./2. * I3_int(M,z,theta,theta2)/(beta*L_int(M,z,theta,theta2))
           + 3.* I4_int(M,z,theta,theta2)/(beta*L_int(M,z,theta,theta2)))

    Eq2 = (1.+fstar)*Sc(c) / gx(c) * (beta*L_int(M,z,theta,theta2)) - Ptot_use(M,z,theta,theta2)

    ans = Eq1**2 + Eq2**2
    return ans

def return_prof_pars(theta2,theta0):
    M,z,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    bounds = [(beta*0.9, beta*1.1), (x_f*0.9, x_f*1.1)]
    ans = fmin(findroots2, theta2, args=(theta0,),xtol=0.001,disp=False)
    #ans = differential_evolution(findroots2, bounds, args=(theta0),disp=False)
    beta_ans, x_f_ans = ans
    rho_0 = rho_0_func(theta0,ans)
    P_0 = P_0_func(theta0,ans,rho_0)
    return P_0, rho_0, x_f_ans


def findroots(theta2,theta0):
    M,z,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(M,z)
    rvir = r200(M,z)
    M_cgs = M*Msol_cgs

    E_inj = Ef * gx(c) * rvir / (Gravity*M_cgs*c) * C_CGS**2

    Eq1 = (3./2. * (K_c(c)*(3.-4.*del_s(c))) - E_inj + 1./3.* Sc(c) / gx(c) * (x_f**3 - c**3)
           - I2_int(M,z,theta,theta2)/L_int(M,z,theta,theta2)
           + 3./2. * I3_int(M,z,theta,theta2)/(beta*L_int(M,z,theta,theta2))
           + 3.* I4_int(M,z,theta,theta2)/(beta*L_int(M,z,theta,theta2)))
    Eq2 = Sc(c) / gx(c) * (beta*L_int(M,z,theta,theta2)) - Ptot_use(M,z,theta,theta2)
    return (Eq1,Eq2)

def find_params_M(M,z,theta_0):
    theta0 = np.append([M,z],[theta_0])
    beta_0 = 1.1
    con_test = con(M,z)
    theta2 = np.array([beta_0 ,con_test*1.01])
    ans = return_prof_pars(theta2,theta0)
    return ans

