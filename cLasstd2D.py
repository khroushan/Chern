""" Class of infinite two-dimensional lattices. 

The module contains hexagonal and square lattices. 
"""
import numpy as np
import numpy.linalg as lg
import FL_funcs as fl
######################################## 

class lattice2D():
    """ class of 2D infinite hexagonal lattice in k-representation.
    """
    def __init__(self, Nd=2, J_hop=1., delta = 1., it = 0, 
                 k_vec = np.zeros((2), float), **kwargs):
        """
        the basic structure of a hexagonal lattice is persumed,
        if no other parameter are not given 
        """
        self.J_hop = J_hop      # the hopping coefficient
        self.Nd = Nd            # dimension of Hamiltonian,
        self.delta = delta      # time-dep hopping amplitude
        self.it = it            # time-interval that system is in
        self.k_vec = k_vec      # the k-vector
        
        # initialize the Hamilnonian matrix
        self.H_kc = fl.H_k(k_vec, it, delta)
    #################### 
    def updateH(self,k_vec,it):
        """
        update the Hamiltonian matrix by new k-vector
        and time-interval it
        input:
        ------
        k_vec: real (2,), 2D (kx,ky) vector
        """
        self.k_vec = k_vec
        self.it = it
        self.H_kc = fl.H_k(k_vec, self.it, self.delta)
    #################### 
    def evolve(self, k_vec, Nt,**kwargs):
        """ evolve the time-dependent parameter of a lattice
        and generate effective Floquet Hamiltonian.

        input:
        ------
        Nt: int, number of intervals in one period
        kwargs: depending on the form of lattice the 
        time-dependent variables can be different. 
        
        return:
        -------
        Efl_k: real (Nd, ) ndarray, sorted quasienergies of 
        effective Floquet Hamiltonian. 
        Ufl_k: complex (Nd,Nd) ndarray, sorted eigenvectors of 
        effective Floquet Hamiltonian
        """
        
        M_eff = np.eye((self.Nd), dtype=complex)   # aux matrix
        T = 1.
        for it in range(Nt):
        
            # update the Hamiltonian for time-inteval
            self.updateH(k_vec, it)

            # return eigenenergies and vectors
            E_k, U = lg.eig(self.H_kc)    

            # U^-1 * exp(H_d) U
            U_inv = lg.inv(U)

            # construct a digonal matrix out of a vector
            M1 = (np.exp(-1.j*E_k*T) * U_inv.T).T

            #MM = np.dot(U_inv,np.dot(H_M, U))
            MM = np.dot(U,M1)
            M_eff = np.dot(M_eff,MM)
        # end of loop
        Ek, Uk = lg.eig( M_eff )
        idx = (np.log(Ek).imag).argsort()
        Efl_k = np.log(Ek).imag[idx]
        Ufl_k = Uk[idx]
        return Efl_k, Ufl_k
    ####################
    def band_structure_static(self, kx_range, ky_range, N_res):
        """
        a Hamiltonian in (kx,ky) is given, evaluate the bandstructire of 
        a static Hamiltonian E(kx,ky)
        input:
        ------
        kx_range: real (2,), the domain of kx
        ky_range: real (2,), the domain of ky
        N_res: int, resolution in kx and kx direction
        
        output:
        -------
        E_arr: real (Nd, N_res, N_res), eigenenergies
        """
        kxR = np.linspace(kx_range[0], kx_range[1], N_res)
        kyR = np.linspace(ky_range[0], ky_range[1], N_res)

        E_arr = np.zeros((2,N_res,N_res), float)
        
        # mesh over area in k-space
#        Kx, Ky = np.meshgrid(kx,ky)

        for ix, kx in enumerate(kxR):
            for iy, ky in enumerate(kyR):
                k_vec = np.array([kx,ky], float)
                # Construct k-representation of Hamiltonian
                self.updateH(k_vec, self.it)

                E_arr[:,ix,iy] = np.sort( np.linalg.eigvalsh(self.H_kc).real )

            #end-loop ky
        #end-loop kx

        return E_arr
    #################### 
    def band_structure_dynamic(self, kx_range, ky_range, N_res):
        """
        a Hamiltonian in (kx,ky,t) is given, evaluate the quasienergy bandstructire
        of a dynamic Hamiltonian E(kx,ky)
        input:
        ------
        kx_range: real (2,), the domain of kx
        ky_range: real (2,), the domain of ky
        N_res: int, resolution in kx and kx direction
        
        output:
        -------
        E_arr: real (Nd, N_res, N_res), eigenenergies
        """
        kxR = np.linspace(kx_range[0], kx_range[1], N_res)
        kyR = np.linspace(ky_range[0], ky_range[1], N_res)

        Nt = 3
        E_arr = np.zeros((2,N_res,N_res), float)
        
        # mesh over area in k-space
#        Kx, Ky = np.meshgrid(kx,ky)

        for ix, kx in enumerate(kxR):
            for iy, ky in enumerate(kyR):
                k_vec = np.array([kx,ky], float)

                # Floquet eigenvalues and eigenenergies
                E_arr[:,ix,iy] , Uaux = self.evolve(k_vec, Nt)

            #end-loop ky
        #end-loop kx

        return E_arr

    ####################
    def latF(self, k_vec, Dk, delta):
        """ Calulating lattice field using the definition:
        F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
        so for each k=(kx,ky) point, four U must be calculate.
        The lattice field has the same dimension of number of
        energy bands.

        input:
        ------
        k_vec=(kx,ky), 
        Dk=(Dkx,Dky), 

        output:
        -------
        F12:lattice field corresponding to each band as a n 
        dimensional vec
        E: Quasienergies
        """

        k = k_vec
        E_sort, psi = lg.eig( fl.H_k(k, self.it, self.delta) )
        E_sort = np.sort(E_sort)

        k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
        E, psiDx = lg.eig( fl.H_k(k, self.it, self.delta) )


        k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
        E, psiDy = lg.eig( fl.H_k(k, self.it, self.delta) )

        k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
        E, psiDxDy = lg.eig( fl.H_k(k, self.it, self.delta) )


        U1x = np.zeros((self.Nd), dtype=complex)
        U2y = np.zeros((self.Nd), dtype=complex)
        U1y = np.zeros((self.Nd), dtype=complex)
        U2x = np.zeros((self.Nd), dtype=complex)

        for i in range(self.Nd):
            U1x[i] = fl.build_U(psi[:,i], psiDx[:,i] )
            U2y[i] = fl.build_U(psi[:,i], psiDy[:,i] )
            U1y[i] = fl.build_U(psiDy[:,i], psiDxDy[:,i] )
            U2x[i] = fl.build_U(psiDx[:,i], psiDxDy[:,i] )

        F12 = np.zeros((self.Nd), dtype=complex)

        F12 = np.log( U1x * U2x * 1./U1y * 1./U2y)

        return F12, E_sort
    ########################################

    def chernNum(self, kx_Bz=np.array([0,4*np.pi/3]), 
                 ky_Bz=np.array([0,2*np.pi/np.sqrt(3)]), 
                 N_res=30):
        """
        To calculate the Chern number of the Hamiltonian (or the Floquet Hamiltonian)
        over its Brillouin zone (BZ). 
        input:
        ------
        kx_Bz: real (2,), kx BZ
        ky_Bz: real (2,), ky BZ
        N_res: int, resolution in kx and kx direction

        output:
        -------
        Chrn_Num: real (Nd,), the Chern number associated with each band.
        """
        x_eps = 0.3                     # shift from Dirac point
        x_res = 20
        kx_int = 0 + x_eps # -np.pi
        kx_fin = 4*np.pi/3 + x_eps
        Dx = (kx_fin - kx_int)/x_res

        y_res = 20
        ky_int = 0 # -np.pi
        ky_fin = 2*np.pi/np.sqrt(3)
        Dy = (ky_fin - ky_int)/y_res

        Nd = self.Nd                          # dimension of the Hamiltonian
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

                LF, E_k = self.latF(k_vec, Dk, self.delta)

                sumN += LF

                # # save data for plotting
                LF_arr[:,ix,iy] = LF.imag

                E_arr[:,ix,iy] = np.sort(E_k.real)

            # End of ky Loop
        # End of kx Loop

        chernN = sumN.imag/(2*np.pi)
        print("Chern number bands are (%.3f, %.3f) "  
              %(chernN[0], chernN[1]))
        print("Sum of all bands Chern Number is %.2f " %(sum(chernN)))
        return chernN, E_arr
        #################### 


