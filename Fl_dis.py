# Program to calculate the Band Structure of Floquet
# operator in a continus honeycomb lattice with stroboscopic
# driven field. When time-dependent field parameter delta=1,
# then one expects the quasienergy spectrum must match the
# bandstructure of graphene layer.

# Author: Amin Ahmdi 
# Date: Dec 18, 2017
# ################################################################
import numpy as np
import numpy.linalg as lg
############################################################

def H_k(k_vec, it, delta=1):
    """construct the k-representation of Hamiltonian of a
    (infinite) hexagonal lattice. In every 1/3 interval of
    one period the hopping amplitudes are changed
    from s to delta*s as cyclic parameter.
    """

    # constants
    s = 1.0                     # hopping amplitude
    
    b = np.zeros((3,2), float)  # unit vectors
    J = np.zeros((3), float)
    
    Hk = np.zeros((2,2), complex)
    
    b[0,:] = np.array([-1/2., np.sqrt(3)/2.], dtype=float)
    b[1,:] = np.array([-1/2.,-np.sqrt(3)/2.], dtype=float)
    b[2,:] = np.array([1./2, 0], dtype=float)

    sigx = np.array([[0,1.],
                     [1.,0]], dtype=complex)

    sigy = np.array([[0,-1.j],
                     [1.j,0]], dtype=complex)

    if (it==0):
        J[0] = delta*s; J[1] = s; J[2] = s
    elif (it==1):
        J[0] = s; J[1] = delta*s; J[2] = s
    elif (it==2):
        J[0] = s; J[1] = s; J[2] = delta*s

    for i in range(3):

        aux1 = np.cos( np.dot(b[i], k_vec) )
        aux2 = np.sin( np.dot(b[i], k_vec) )
        Hk += -J[i] * ( aux1*sigx + aux2*sigy )

    return Hk
###########################################################
def H_eff(k_vec, delta):
    """Construct the Floquet effective Hamiltonian by evolving
    of the Hamiltonian in every 1/3 period interval"""

    M_eff = np.eye((NN), dtype=complex)   # aux matrix
    for it in range(N_t):
        
        # Construct Fourier transform of Hamiltonian
        # and diagonalization
        H_kc = H_k(k_vec, it, delta)
        
        # return eigenenergies and vectors
        E_k, U = lg.eig(H_kc)    
        
        # U^-1 * exp(H_d) U
        U_inv = lg.inv(U)
        
        # construct a digonal matrix out of a vector
        #H_M= np.diag(np.exp((-1j/3.)*E_k*T))
        M1 = (np.exp((-1j/3.)*E_k*T) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)

    return M_eff
############################################################

############################################################
##############         Main Program     ####################
############################################################
N_k = 20                                # Num of k-points
N_t = 3                                  # Num of time intervals
NN = 2
Nd = 2                                   # dimension
T = 1.                                   # One period of driven field

# During each iterval T/N_t, the Hamiltonian is time-independent
H_kc = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=float)      # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors

# different hopping amplitude
delta = input("Enter the time-parameter hopping coefficient: ") 
J = np.pi/16.                             # hopping amplitude 
data_plot = np.zeros((N_k, N_k,2), dtype=float)


# loop over k, first BZ 
for ikx in range(N_k):
    kx = -np.pi +  ikx*(2*np.pi/N_k)
    # ka = ik*(np.pi/N_k)
    for iky in range(N_k):
        ky = -np.pi +  iky*(2*np.pi/N_k)
        k_vec = np.array([kx,ky], float)
    
        M_eff = H_eff(k_vec, delta)
        
        E_Fl, UF = lg.eig(M_eff)
        E_k = np.sort(np.log(E_Fl).imag)
        data_plot[ikx,iky,:] = E_k/T

############################################################
# save data
np.save("./Result/Fl_ddp", data_plot)

