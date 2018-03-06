# This program calculate the chern number of a 2D system using
# technique that was introduced in paper by T. Fukui et. al.

# Author: Amin Ahmadi
# Date(in): Oct 30, 2017
# Date2: Nov 6, 2017
# This version would be more structured
############################################################
# importing numpy and linear algebra modules
import numpy as np
import numpy.linalg as lg

########################################
###            Functions             ###
########################################
def H_k(k_vec, dim=3):
    """function to construct the Hamiltonian of a 2DEG in
    presence of an applied magnetic field. The magnetic
    field is introduced using Landau's guage.  
    input:
    ------
    k_vec: vec(2), float, (kx, ky)
    dim: integer, dimension of Hamiltonian, depends on magnetic flux

    return:
    -------
    Hk: (2,2) complex, k-representation of H

    """
    Hk = np.zeros((dim,dim), dtype=complex)
    t = 1                       # hopping amplitude
    phi = 1/3.                  # flux per plaquette

    kx = k_vec[0]
    ky = k_vec[1]
    # diagonal elements
    for i in range(3):
        Hk[i,i] = -2*t*np.cos( ky - 2.*(i+1)*np.pi*phi )

    # off-diagonal elements
    Hk[0,1] = -t
    Hk[1,2] = -t
    Hk[0,2] = -t*np.exp(-3.j*kx)

    # Make it hermitian
    Hk = Hk + Hk.conj().T

    return Hk
############################################################

def build_U(vec1,vec2):
    """function to calculate the iner product of two
    eigenvectors divided by the norm:
    
    U = <psi|psi+mu>/|<psi|psi+mu>|

    input:
    ------
    vec, vec2: vectors complex.

    return:
    -------
    U: scalar complex number

    """

    # U = <psi|psi+mu>/|<psi|psi+mu>|
    in_product = np.dot(vec1,vec2.conj())

    U = in_product / np.abs(in_product)

    return U
############################################################

def latF(k_vec, Dk, dim):
    """calulate lattice field using the definition: F12 = ln[
    U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ] for each
    k=(kx,ky) point, four U must be calculated.  The lattice
    field has the same dimension of the number of energy
    bands.
    
    input:
    ------
    k_vec:vec(2), float, (kx,ky).
    Dk: vec(2), float, (Dkx,Dky),
    dim:integer,  dim of H(k)
    
    return:
    -------
    F12:vec(dim), complex, lattice field corresponding to each band.
    E_sort: vec(dim) float, eigenenergies.
    """

    # Here we calculate the band structure and sort
    # them from low to high eigenenergies

    k = k_vec
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    E_sort = E[idx]
    psi = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDx = aux[:,idx]

    k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDy = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDxDy = aux[:,idx]

    U1x = np.zeros((dim), dtype=complex)
    U2y = np.zeros((dim), dtype=complex)
    U1y = np.zeros((dim), dtype=complex)
    U2x = np.zeros((dim), dtype=complex)

    for i in range(dim):
        U1x[i] = build_U(psi[:,i], psiDx[:,i] )
        U2y[i] = build_U(psi[:,i], psiDy[:,i] )
        U1y[i] = build_U(psiDy[:,i], psiDxDy[:,i] )
        U2x[i] = build_U(psiDx[:,i], psiDxDy[:,i] )

    F12 = np.zeros((dim), dtype=complex)

    F12 = np.log( U1x * U2x * 1./U1y * 1./U2y)

    return F12, E_sort

##################################################
###             Main program                   ###
##################################################

x_res = 50
y_res = 50
Nd = 3                          # dimension of the Hamiltonian

Dx = (2.*np.pi/3.)/x_res
Dy = (2.*np.pi)/y_res
Dk = np.array([Dx,Dy], float)

LF = np.zeros((Nd), dtype=complex)
LF_arr = np.zeros((Nd,x_res, y_res), dtype=float) # plotting array
sumN = np.zeros((Nd), dtype=complex)
E_k = np.zeros((Nd), dtype=complex)
chernN = np.zeros((Nd), dtype=complex)

for ix in range(x_res):

    kx = ix*Dx
    for iy in range(y_res):

        ky = iy*Dy

        k_vec = np.array([kx,ky], float)
        LF, E_k = latF(k_vec, Dk, Nd)

        sumN += LF

        # save data for plotting
        LF_arr[:,ix,iy] = -LF.imag/(2.*np.pi) 

chernN = sumN.imag/(2.*np.pi)
print("Chern number associated with each band: ", chernN)

##################################################
###             Main program                   ###
##################################################

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = pl.figure()
ax = fig.gca(projection='3d')

kx = np.linspace(0,2.*np.pi/3., x_res)
ky = np.linspace(0,2.*np.pi, y_res)

kx, ky = np.meshgrid(ky,kx)

surf = ax.plot_wireframe(ky, kx, LF_arr[1,:,:], rstride=1, cstride=1, color='0.4')
# ax.set_xlim(0,2.*np.pi/3.)

# Set viewpoint.
ax.azim = -60
ax.elev = 30

# Label axes.
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_xticks([0.0, np.pi/3, 2*np.pi/3])
ax.set_xticklabels([r'$0$', r'$\pi/3$', r'$2\pi/3$'], fontsize=16)
ax.set_xlim(0,2*np.pi/3)

ax.set_ylabel(r'$k_y$', fontsize=18)
ax.yaxis._axinfo['label']['space_factor'] = 2.5
ax.set_yticks([0.0, np.pi, 2*np.pi])
ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=16)
ax.set_ylim(0,2*np.pi)


ax.set_zlabel(r'$i\tilde{F}_{12}$', fontsize=18)
ax.zaxis._axinfo['label']['space_factor'] = 2.5

# ax.set_zticks([""])

# ax.set_zticklabels([""])


# surf = ax.plot_surface(ky, kx, LF_arr[1,:,:], rstride=1, cstride=1, color='g', norm=0.1, shade=True,
#                           facecolor='b', linewidth=0, antialiased=False) #cmap=cm.jet

# To rescale the plot
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 1.5, 1, 1]))

# ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
# ax.pbaspect = [.6, 2.6, 0.25]
# fig.colorbar(surf, shrink=1., aspect=5)

pl.show()
# fig.savefig("chr3.pdf", bbox_inches='tight')
