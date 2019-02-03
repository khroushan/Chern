# File: FL_funcs.py
# Functions that commonly used in Hamiltonian construction and
# Chern number calculation are collected here

# Author: Amin Ahmadi
# Date: Jan 31, 2018
############################################################
import numpy as np
import numpy.linalg as lg
############################################################
def H_k(k_vec, it, delta=1):
    """ construct the k-representation of Hamiltonian of a 
    hexagonal lattice. In every third interval of one period 
    the hopping amplitude different from the other two.
    
    input:
    ------
    k_vec: real, 2D (kx,ky) vector
    it: integer, the time step for periodic driven field
    delta: real, time-dependent hopping amplitude

    return:
    -------
    H_k: 2x2 matrix of the effective Hamiltonian
    """
    
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
    
    input: 
    ------
    k_vec: real, 2D (kx,ky) vector
    delta: time-dependent hopping parameter

    return:
    -------
    Efl_k: real (2,) ndarray, quasienergies of
    effective Floquet Hamiltonian
    Ufl_k: complex (2,2) ndarray, eigenvectors of
    effective Floquet Hamiltonian
    """

    Nd = 2                              # dimension of Hamiltonian
    N_t = 3                             # number of time interval
    T = 1                               # period of field
    
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
    # end of loop
    E, aux = lg.eig( M_eff )
    idx = (np.log(E).imag).argsort()
    Efl_k = np.log(E).imag[idx]
    Ufl_k = aux[:,idx]

    return Efl_k, Ufl_k
############################################################

def build_U(vec1,vec2):

    """ This function calculate the iner product of two
    eigenvectors divided by the norm: 
    
    U = <psi|psi+mu>/|<psi|psi+mu>|

    input:
    ------
    vec1
    vec2
    
    return: 
    -------
    scalar complex number
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

    k = k_vec
    E_sort, psi = H_eff(k, delta)
    
    k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
    E, psiDx = H_eff(k, delta)
    
    
    k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
    E, psiDy = H_eff(k, delta)
    
    k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
    E, psiDxDy = H_eff(k, delta)
    
    
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
