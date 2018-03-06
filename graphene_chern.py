# This program calculate the chern number of a 2D graphene 
# technique that was introduced in paper by T. Fukui et. al.

# Author: Amin Ahmadi
# Date: Dec 20, 2017
# Date: Dec 26, 2017     Make sure the Dirac points are at right places
# Date: Jan 9, 2017      TRS breaking term is added and result are corrects.
############################################################
# importing numpy and linear algebra modules
import numpy as np
import numpy.linalg as lg

def Gamma(kx,ky):
    """Returns the gamma function associated with a Hamiltonian
    of a 2D graphene layer.
    
    in: kx, ky
    out: scalar complex function
    """
    gamma = 2*np.exp(1.j*kx/2)*np.cos(np.sqrt(3.)*ky/2) + \
    np.exp(-1.j*kx)

    return gamma
############################################################

def Beta(kx,ky):
    """Returns the beta function associated with the next nearest 
    hopping between the same sites A to A and B to B in graphen. The 
    hopping amplitude is purely imaginary.

    This term is Haldane model to break the TRS in a graphene layer.
        
    in: kx, ky
    out: scalar complex function
    """
    k_vec = np.array([kx,ky], float)
    b1 = np.array([0,np.sqrt(3)], float)
    b2 = (-3./2)*np.array([1,1./np.sqrt(3)], float)
    b3 = (3./2)*np.array([1,-1./np.sqrt(3)], float)
    
    beta = np.sin(np.dot(k_vec,b1)) + np.sin(np.dot(k_vec,b2)) + \
           np.sin(np.dot(k_vec,b3))

    return beta
############################################################
def H_k(k_vec):

    """This function gives the matrix 2x2 of Hamiltonian of an
    infinite graphene layer
    input:
    ------
    k_vec: 2x1 matrix float,  (kx, ky)
    
    return:
    -------
    Hk: 2x2 complex, k-representation Hamiltonian
    """
    t1=1.
    m=0.5
    t2=  m/(3*np.sqrt(3)) - 0.2
    t_so= 0.001
    
    kx = k_vec[0]
    ky = k_vec[1]
    Nd = 4                  # including spin-degree of freedom
    phi1 = np.sqrt(3) * np.exp(0.5j*kx) * \
           np.sin(np.sqrt(3)*0.5*ky)
    phi2 = np.exp(-1.j*kx) - np.exp(.5j*kx) * \
           np.cos(np.sqrt(3)*0.5*ky)



    Hk = np.zeros((Nd,Nd), dtype=complex)

    
    Hk[0,0] = 0.5* ( m +  t2*Beta(kx,ky) ) # To break TRS
    Hk[0,1] = t1*Gamma(kx,ky)
    Hk[0,3] = t_so  * (1.j*phi1 + phi2)
    

    Hk[1,1] = - Hk[0,0]
    Hk[1,2] = t_so  * (-1.j*phi1.conj() - phi2.conj())

    Hk[2,2] = Hk[0,0]
    Hk[2,3] = Hk[0,1]

    Hk[3,3] = - Hk[0,0]
    
    Hk += Hk.T.conj()

    return Hk
############################################################
def H_Rashba(k_vec):
    
    """This function returns the 4x4 Hamiltonian of a graphene
    layer, including the Haldane next nearest neighbor hopping
    and Rashba SOI.
    
    input:
    ------
    k_vec: vec(float,float), (kx,ky)
    t2: float, hoping coefficient to the next nearest neighbor
    ts: float, Rashba SOI strength

    return:
    -------
    Hk: 4x4 complex matrix, Hamiltonian
    """
    m=2
    tso= 0.06# m/(3*np.sqrt(3)) - 1
    tR=0.0000001
    
    kx = k_vec[0]
    ky = k_vec[1]
    Nd = 4                  # including spin-degree of freedom

    gamma = np.exp(1.j*kx) + 2.* np.exp(-0.5j*kx) * \
            np.cos(np.sqrt(3)*0.5*ky)
    phi1 = np.exp(1.j*kx) + 2.* np.exp(-0.5j*kx) * \
           np.cos(np.sqrt(3)*0.5*ky + 2.*np.pi/3)
    phi2 = np.exp(1.j*kx) + 2.* np.exp(-0.5j*kx) * \
           np.cos(np.sqrt(3)*0.5*ky - 2.*np.pi/3)



    Hk = np.zeros((Nd,Nd), dtype=complex)

    
    Hk[0,0] = 0.5* ( m +  2*tso*Beta(kx,ky) ) # To break TRS
    Hk[0,1] = gamma
    Hk[0,3] = tR  * (1.j*phi1)
    

    Hk[1,1] = Hk[0,0]
    Hk[1,2] = tR  * (-1.j*phi2.conj())

    Hk[2,2] = - Hk[0,0]
    Hk[2,3] = Hk[0,1]

    Hk[3,3] = - Hk[0,0]
    
    Hk += Hk.T.conj()

    return Hk
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

def latF(k_vec, Dk, dim):
    """ Calulating lattice field using the definition:
    F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
    so for each k=(kx,ky) point, four U must be calculate.
    The lattice field has the same dimension of number of
    energy bands.
    
    in: k-point k_vec=(kx,ky), Dk=(Dkx,Dky), dim: dim of H(k)
    out: lattice field corresponding to each band as a n 
    dimensional vec
    """

    # Here we calculate the band structure and sort
    # them from low to high eigenenergies

    k = k_vec
    E, aux = lg.eig( H_Rashba(k) )
    idx = E.real.argsort()
    E_sort = E[idx].real
    psi = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
    E, aux = lg.eig( H_Rashba(k) )
    idx = E.real.argsort()
    psiDx  = aux[:,idx]

    k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_Rashba(k) )
    idx = E.real.argsort()
    psiDy = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_Rashba(k) )
    idx = E.real.argsort()
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

######################################## 
##########   Main Program   ############
########################################
x_eps = 0.3                     # shift from Dirac point
x_res = 20
kx_int = 0 + x_eps # -np.pi
kx_fin = 4*np.pi/3 + x_eps
Dx = (kx_fin - kx_int)/x_res

y_res = 20
ky_int = 0 # -np.pi
ky_fin = 2*np.pi/np.sqrt(3)
Dy = (ky_fin - ky_int)/y_res

Nd = 4                          # dimension of the Hamiltonian
Dk = np.array([Dx,Dy], float)

LF = np.zeros((Nd), dtype=complex)
LF_arr = np.zeros((Nd,x_res, y_res), dtype=float)
E_arr = np.zeros((Nd,x_res, y_res), dtype=float)
sumN = np.zeros((Nd), dtype=complex)
E_k = np.zeros((Nd), dtype=complex)
chernN = np.zeros((Nd), dtype=complex)

# Loop over kx
for ix in range(x_res):
    kx = kx_int + ix*Dx
    
    # Loop over ky
    for iy in range(y_res):
        ky = ky_int + iy*Dy

        k_vec = np.array([kx,ky], float)
        
        LF, E_k = latF(k_vec, Dk, Nd)

        sumN += LF

        # # save data for plotting
        LF_arr[:,ix,iy] = LF.imag

        E_arr[:,ix,iy] = np.sort(E_k.real)

    # End of ky Loop
# End of kx Loop

chernN = sumN.imag/(2*np.pi)
print("Chern number bands are (%.3f, %.3f, %.3f, %.3f) "  
      %(chernN[0], chernN[1], chernN[2], chernN[3]))
print("Sum of all bands Chern Number is %.2f " %(sum(chernN)))

######################################## 
##########       Plot       ############
########################################
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
#  Dirac points are
K1 = 2*np.pi/3*np.array([1., 1./np.sqrt(3)], dtype=float)
K2 = 2*np.pi/3*np.array([1., -1./np.sqrt(3)], dtype=float)



fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1)

ext = [kx_int, kx_fin, ky_int, ky_fin] 
im = ax.imshow(E_arr[1,:,:].T, extent=ext, 
               origin='lower',  cmap=pl.cm.RdBu) 
cset = ax.contour(E_arr[0,:,:].T, np.arange(-3,0,0.3),
                  origin='lower', extent=ext, 
                  linewidths=2,cmap=pl.cm.Set2)
ax.plot(K1[0], K1[1], '*k', label=r'$K_1$')
ax.set_xlim(kx_int, kx_fin)
ax.set_ylim(ky_int, ky_fin)

ax.legend()

# # adding the Contour lines with labels
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right

ax2 = fig.add_subplot(1,2,2, projection='3d')


kx = np.linspace(kx_int,kx_fin, x_res)
ky = np.linspace(ky_int,ky_fin, y_res)

kx, ky = np.meshgrid(kx,ky)

surf = ax2.plot_wireframe(kx, ky, LF_arr[0,:,:].T, 
                          rstride=1, cstride=1, color='0.4')
# # ax.set_xlim(0,2.*np.pi/3.)

ax2.set_xlabel(r'$k_x$')
ax2.set_ylabel(r'$k_y$')

pl.show()
