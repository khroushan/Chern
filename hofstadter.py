# This program calculate the eigen energies of a qantum Hall
# system (square lattice), and plot the energy as a function
# of flux (applied magnetic filed). The plot of nergy vs. flux
# would generate hofstadter butterfly

# Author: Amin Ahmadi
# Date: Jan 16, 2018
############################################################
# importing numpy and linear algebra modules
import numpy as np
import numpy.linalg as lg

# Function that calculate Hamiltonian H(k)
def H_k(k_vec, p, q):
    """ This function calulate the Hamiltonian of a 
    2D electron gas in presence of an applied magnetic
    field. The magnetic field is introduced using 
    Landau's guage.
    in: kx, ky, 
    p and q are the ratio of p/q = phi/phi0
    out: H_k, k-representation of H 
    """
    Hk = np.zeros((q,q), dtype=complex)
    t = 1.                            # hopping amplitude
    phi_ratio = (1.*p)/q              # flux per plaquette

    kx = k_vec[0]
    ky = k_vec[1]
    # diagonal elements
    for i in range(q):
        Hk[i,i] = -t*np.cos( ky - 2.*(i+1)*np.pi*phi_ratio )

    # off-diagonal elements
    for i in range(q-1):
        Hk[i,i+1] = -t

    # Magnetic Bloch element
    Hk[0,q-1] = -t*np.exp(-q*1.j*kx)

    # Make it hermitian
    Hk = Hk + Hk.conj().T

    return Hk

############################################################
###################  Main Program  #########################
############################################################

q = 501                       # phi/phi0 = p/q 
k_vec = np.array([3.0,0], float)
E_arr = np.zeros((q,q), float)
for p in range(q):

    
        E_k= lg.eigvalsh(H_k(k_vec, p, q))
        
        # save data for plotting
        E_arr[:, p] = E_k[:]

# save date in file
np.save('./Result/hofs_501_00', E_arr)

############################################################
####################   plotting    #########################
############################################################

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

pl.rc("font", family='serif')
pl.rc('text', usetex=True)

fig = pl.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)

pq = np.linspace(0,1,q)

for p in range(q):
    ax.plot(pq, E_arr[p,:],',k')

ax.set_xlabel(r'$\frac{\phi}{\phi_0}$', fontsize=16)
ax.set_ylabel(r'$E$', fontsize=16)
    
pl.show()
fig.savefig('./Result/butterfly_501.pdf', bbox_inches='tight')
