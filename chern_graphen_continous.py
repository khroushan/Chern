# Program to calculate the Band Structure of Floquet
# operator and the chern number associated with quasi
# bandstructure in a continus honeycomb lattice with
# stroboscopic driven field

# Author: Amin Ahmdi
# Date: Dec 18, 2017
# Date: Jan 8, 2018
# Date: Jan 10, 2018      Code is fixed and result are correct
# ################################################################
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as pl

############################################################
def H_k(k_vec, it, delta=1):
    """ construct the k-representation of Hamiltonian of a 
    hexagonal lattice. In every third interval of one period 
    the hopping amplitude different from the other two"""

    # constants
    s = np.pi/16                       # hopping amplitude
    m = 0.0                            # Dirac mass parameter
    
    b = np.zeros((3,2), float)         # nearest neighbors unit vectors
    J = np.zeros((3), float)           # nearest neighbors hopping
    
    Hk = np.zeros((2,2), complex)      # Hamiltonian
    
    b[0,:] = 1./2*np.array([1, np.sqrt(3)], dtype=float)
    b[1,:] = 1./2*np.array([1,-np.sqrt(3)], dtype=float)
    b[2,:] = -np.array([1., 0], dtype=float)

    # Pauli matrices
    
    sigx = np.array([[0,1.],
                     [1.,0]], dtype=complex)

    sigy = np.array([[0,-1.j],
                     [1.j,0]], dtype=complex)

    sigz = np.array([[1.,0],
                     [0,-1.]], dtype=complex)

    # Time-interval
    if (it==0):
        J[0] = delta*s; J[1] = s; J[2] = s
    elif (it==1):
        J[0] = s; J[1] = delta*s; J[2] = s
    elif (it==2):
        J[0] = s; J[1] = s; J[2] = delta*s

    for i in range(3):

        aux1 = np.cos( np.dot(b[i], k_vec) )
        aux2 = np.sin( np.dot(b[i], k_vec) )
        Hk += -J[i] * ( aux1*sigx - aux2*sigy )

    Hk += m*sigz
    return Hk
###########################################################
def H_eff(k_vec, delta):
    """Calculate the effective Floquet Hamiltonian using
    evolution for three time interval.
    
    in: 
    k_vec: momentum vector 
    delta: time-dependent hopping parameter

    out:
    exp(iH_eff T)
    """

    Nd = 2                              # dimension of Hamiltonian
    N_t = 3                             # number of time interval
    
    M_eff = np.eye((Nd), dtype=complex)   # aux matrix
    for it in range(N_t):
        
        # Construct Fourier transform of Hamiltonian
        H_kc = H_k(k_vec, it, delta)
        
        # return eigenenergies and vectors
        E_k, U = lg.eig(H_kc)    
        
        # U^-1 * exp(H_d) U
        U_inv = lg.inv(U)
        
        # construct a digonal matrix out of a vector
        M1 = (np.exp(-1.j*E_k*T) * U_inv.T).T

        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)

    return M_eff
############################################################

def build_U(vec1,vec2):

    """ This function calculate the iner product of two
    eigenvectors divided by the norm: 
    
    U = <psi|psi+mu>/|<psi|psi+mu>|

    in: two vectors vec1, and vec2
    out: scalar complex number
    """

    # U = <psi|psi+mu>/|<psi|psi+mu>|
    in_product = np.dot(vec1,vec2.conj())

    U = in_product / np.abs(in_product)

    return U
############################################################

def latF(k_vec, Dk, delta, dim=2):
    """ Calulating lattice field using the definition:
    F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
    so for each k=(kx,ky) point, four U must be calculate.
    The lattice field has the same dimension of number of
    energy bands.
    
    in: 
    k_vec=(kx,ky), 
    Dk=(Dkx,Dky), 
    dim: dim of H(k)
    
    out: 
    F12:lattice field corresponding to each band as a n 
    dimensional vec
    E: Quasienergies
    """

    # Here we calculate the band structure and sort
    # them from low to high eigenenergies

    k = k_vec
    H_k = H_eff(k, delta)
    E, aux = lg.eig( H_k )
    idx = (np.log(E).imag).argsort()
    E_sort = np.log(E).imag[idx]
    # E_sort = np.log(E).imag
    psi = aux[:,idx]
    
    k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
    H_k = H_eff(k, delta)
    E, aux = lg.eig( H_k )
    idx = (np.log(E).imag).argsort()
    psiDx = aux[:,idx]
    
    k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
    H_k = H_eff(k, delta)
    E, aux = lg.eig( H_k )
    idx = (np.log(E).imag).argsort()
    psiDy = aux[:,idx]
    
    k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
    H_k = H_eff(k, delta)
    E, aux = lg.eig( H_k )
    idx = (np.log(E).imag).argsort()
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

############################################################
##############         Main Program     ####################
############################################################

# The k-domain is the brillouin zone

x_eps = 0.0                     # to scape the Dirac point at the edge of integrating area
x_res = 10                      # kx resolution
kx_int = 0 + x_eps              # initial point
kx_fin = 4*np.pi/3 + x_eps      # final point
Dx = (kx_fin - kx_int)/x_res

y_res = 10
ky_int = 0
ky_fin = 2*np.pi/np.sqrt(3)
Dy = (ky_fin - ky_int)/y_res

Nd = 2                          # dimension of the Hamiltonian
Dk = np.array([Dx,Dy], float)

T = 1.                          # One period of driven field


H_kc = np.zeros((Nd,Nd), dtype=complex)   # k-representation H
E_k = np.zeros((Nd), dtype=complex)       # eigenenergies
psi_k = np.zeros((Nd,Nd), dtype=complex)  # matrix of eigenvectors

# different hopping amplitude
delta = input("Enter the hopping difference coefficient: ")

# array to save data for plotting
E_arr = np.zeros((Nd, x_res, y_res), dtype=float)
LF_arr = np.zeros((Nd, x_res, y_res), dtype=float)

LF = np.zeros((Nd), dtype=complex)
sumN = np.zeros((Nd), dtype=complex)
E_k = np.zeros((Nd), dtype=complex)

# Chern number associated with each band
chernN = np.zeros((Nd), dtype=complex)




# loop over kx, first BZ 
for ix in range(x_res):
    kx = kx_int + ix*Dx

    # loop over ky, first BZ 
    for iy in range(y_res):
        ky = ky_int +  iy*Dy

        k_vec = np.array([kx,ky], float)
    

        LF, E_k = latF(k_vec, Dk, delta, Nd)
        
        E_arr[:,ix,iy] = E_k/T

        sumN += LF

        # save data for plotting
        LF_arr[:,ix,iy] = -LF.imag/(2.*np.pi) 

chernN = sumN.imag/(2.*np.pi)
print chernN
print sum(chernN)
############################################################
# save data
# np.save("Fl_ddpi4", data_plot)
######################################## 
##########       Plot       ############
########################################
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

# plot1
########################################
fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1)

ext = [kx_int, kx_fin, ky_int, ky_fin] 
im = ax.imshow(E_arr[1,:,:].T, extent=ext, origin='lower',  cmap=pl.cm.RdBu) 
cset = ax.contour(E_arr[0,:,:].T, np.arange(-3,0,0.3), origin='lower', 
                  extent=ext, linewidths=2,cmap=pl.cm.Set2)

# plot2
########################################
ax2 = fig.add_subplot(1,2,2, projection='3d')


kx = np.linspace(kx_int,kx_fin, x_res)
ky = np.linspace(ky_int,ky_fin, y_res)

kx, ky = np.meshgrid(kx,ky)

surf = ax2.plot_wireframe(kx, ky, LF_arr[0,:,:].T, rstride=1, cstride=1, color='0.4')
# # ax.set_xlim(0,2.*np.pi/3.)

ax2.set_xlabel(r'$k_x$')
ax2.set_ylabel(r'$k_y$')

pl.show()

