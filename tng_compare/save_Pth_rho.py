import sys
sys.path.append("..")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mopc
from mopc.params import cosmo_params
from mopc.cosmo import rho_cz
from mopc.two_halo import r200m_kpc, r200c_kpc, r200t_kpc
from mopc import two_halo as two

from mpi4py import MPI
# mpirun -np 22 python save_Pth_rho.py 22 0.3; mpirun -np 22 python save_Pth_rho.py 22 0.5
myrank = MPI.COMM_WORLD.Get_rank()
n_ranks = int(sys.argv[1])

# constants
Msol_cgs = 1.989e33
G_cgs = 6.67259e-8 #cm3/g/s2

# construct mass bins
logbins = np.linspace(12, 14.6, 23)
logbinc = (logbins[1:]+logbins[:-1])*.5
mbinc = 10.**logbinc # Msun

# split the mass bins
n_jump = len(mbinc)//n_ranks
assert len(mbinc) % n_ranks == 0
assert len(mbinc) // n_ranks == 1
mbinc = mbinc[myrank*n_jump: (myrank+1)*n_jump]

# r/rhalo bins
x = np.logspace(-3., 1., 50)

# choose redshift
z = float(sys.argv[2]) #0.5 # 0.3

# rho baryons critical (thought it should be rho_b = fb rho_m = fb omega_m rho_crit)
#fb = cosmo_params['Omega_b']/cosmo_params['Omega_m']
#rhogas_c = fb * rho_cz(z) #g/cm3
#P200m = G_cgs * m*Msol_cgs * 200. * rho_cz(z) * fb /(2.*r200(m,z))

# initialize final arrays
want_rho = True # pick between Pth and rho
if want_rho:
    rho_arr = np.zeros((len(x), 2, len(mbinc)))
else:
    Pth_arr = np.zeros((len(x), 2, len(mbinc)))
    
# initialize virial radius array
r200m_kpcs = np.zeros(len(mbinc))
r200c_kpcs = np.zeros(len(mbinc))
r200t_kpcs = np.zeros(len(mbinc))

# loop over halo masses
for k in range(len(mbinc)):
    # mass and virial radius
    m = mbinc[k]
    r200m_kpcs[k] = r200m_kpc(m, z)
    r200c_kpcs[k] = r200c_kpc(m, z)
    r200t_kpcs[k] = r200t_kpc(m, z)
    print(f"mass = {m:.2e}", k)
    sys.stdout.flush()
    
    if want_rho:
        # one-halo density
        rho1h = two.rho_gnfw(x, m, z)  #g/cm3

        # two-halo density
        rho2h = []
        for ix in range(len(x)):
            rho2h.append(two.rho_2h(x[ix], m, z))
        rho2h = np.array(rho2h)  #g/cm3

        rho_arr[:, 0, k] = rho1h
        rho_arr[:, 1, k] = rho2h

    else:
        # one-halo pressure
        Pth1h = two.Pth_gnfw(x, m, z) #[g/cm/s2]
        print("done with one-halo")
        sys.stdout.flush()
    
        # two-halo pressure
        Pth2h = []
        for ix in range(len(x)):
            Pth2h.append(two.Pth_2h(x[ix], m, z))
        Pth2h = np.array(Pth2h)  #g/cm3
        print("done with two-halo")
        sys.stdout.flush()
    
        Pth_arr[:, 0, k] = Pth1h
        Pth_arr[:, 1, k] = Pth2h
    
    
# save arrays
if want_rho:
    np.savez(f"data/rhoGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz", rho_arr=rho_arr, rbinc_ratio=x, r200m_kpcs=r200m_kpcs, r200c_kpcs=r200c_kpcs, r200t_kpcs=r200t_kpcs, mbinc_Msuns=mbinc)
else:
    np.savez(f"data/PthGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz", Pth_arr=Pth_arr, rbinc_ratio=x, r200m_kpcs=r200m_kpcs, r200c_kpcs=r200c_kpcs, r200t_kpcs=r200t_kpcs, mbinc_Msuns=mbinc)
