import sys
import os
import numpy as np

"""
python combine_Pth_rho.py 22 0.3
python combine_Pth_rho.py 22 0.5

"""

# choices
want_rho = False
n_ranks = int(sys.argv[1])
z = float(sys.argv[2]) # 0.3, 0.5

# construct mass bins
logbins = np.linspace(12, 14.6, 23)
logbinc = (logbins[1:]+logbins[:-1])*.5
mbinc = 10.**logbinc # Msun
assert len(mbinc) == n_ranks

# r/rvir bins
x = np.logspace(-3., 1., 50) # same as rbinc_ratio, could grab that info from 1st file instead

# init arrays
Pth_arr = np.zeros((len(x), 2, len(mbinc)))
r200m_kpcs = np.zeros(len(mbinc))
r200c_kpcs = np.zeros(len(mbinc))
r200t_kpcs = np.zeros(len(mbinc))
mbinc_Msuns = np.zeros(len(mbinc))

for myrank in range(n_ranks):
    if want_rho:
        data = np.load(f"data/rhoGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz")
        Pth_arr[:, :, myrank] = data['rho_arr'][:, :, 0] # because there is only one dimension to this if nranks is mbinc
    else:
        data = np.load(f"data/PthGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz")
        Pth_arr[:, :, myrank] = data['Pth_arr'][:, :, 0] # because there is only one dimension to this if nranks is mbinc
    #print(f"{mbinc[myrank]:.2e}")
    r200m_kpcs[myrank] = data['r200m_kpcs']
    r200c_kpcs[myrank] = data['r200c_kpcs']
    r200t_kpcs[myrank] = data['r200t_kpcs']
    mbinc_Msuns[myrank] = data['mbinc_Msuns']
rbinc_ratio = data['rbinc_ratio']

if want_rho:
    np.savez(f"data/rhoGNFW_z{z:.1}.npz", rho_arr=Pth_arr, rbinc_ratio=rbinc_ratio, r200m_kpcs=r200m_kpcs, r200c_kpcs=r200c_kpcs, r200t_kpcs=r200t_kpcs, mbinc_Msuns=mbinc_Msuns)
else:
    np.savez(f"data/PthGNFW_z{z:.1}.npz", Pth_arr=Pth_arr, rbinc_ratio=rbinc_ratio, r200m_kpcs=r200m_kpcs, r200c_kpcs=r200c_kpcs, r200t_kpcs=r200t_kpcs, mbinc_Msuns=mbinc_Msuns)

for myrank in range(n_ranks):
    if want_rho:
        os.unlink(f"data/rhoGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz")
    else:
        os.unlink(f"data/PthGNFW_z{z:.1}_rank{myrank:d}_{n_ranks:d}.npz")
