# Program to calculate the Band Structure of Floquet
# operator and the chern number associated with quasi
# bandstructure in a continus honeycomb lattice with
# stroboscopic driven field

# Author: Amin Ahmdi
# Date: Dec 18, 2017
# Date: Jan 8, 2018
# Date: Jan 10, 2018      Code is fixed and result are correct
# Date: Jan 30, 2018      Separate general functions into another file 
# ################################################################
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as pl

# Internal package for Chern Num calculation
import FL_funcs as FL

############################################################
##############         Main Program     ####################
############################################################

# The k-domain is the brillouin zone

x_res = 30                     # kx resolution
kx_int = 0                      # initial point
kx_fin = 4*np.pi/3              # final point
Dx = (kx_fin - kx_int)/x_res

y_res = 30
ky_int = 0
ky_fin = 2*np.pi/np.sqrt(3)
Dy = (ky_fin - ky_int)/y_res

Nd = 2                          # dimension of the Hamiltonian
Dk = np.array([Dx,Dy], float)

T = 1.                          # One period of driven field


H_kc = np.zeros((Nd,Nd), dtype=complex)   # k-representation H
E_k = np.zeros((Nd), dtype=complex)       # eigenenergies
psi_k = np.zeros((Nd,Nd), dtype=complex)  # matrix of eigenvectors

# time-dependent hopping amplitude
delta = float(input("Enter the hopping difference coefficient: "))

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
    

        LF, E_k = FL.latF(k_vec, Dk, delta, Nd)
        
        E_arr[:,ix,iy] = E_k/T

        sumN += LF

        # save data for plotting
        LF_arr[:,ix,iy] = -LF.imag/(2.*np.pi) 

chernN = sumN.imag/(2.*np.pi)
print(chernN)
print(sum(chernN))
############################################################
# save data
np.save("./Result/Fl_E4", E_arr)
np.save("./Result/Fl_F124", LF_arr)
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

surf = ax2.plot_wireframe(kx, ky, LF_arr[0,:,:].T, rstride=1,
                          cstride=1, color='0.4')
# # ax.set_xlim(0,2.*np.pi/3.)

ax2.set_xlabel(r'$k_x$')
ax2.set_ylabel(r'$k_y$')

pl.show()

