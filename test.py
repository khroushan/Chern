# test file for class: classtd2D, file: 'classtd2D.py'
import numpy as np
import cLasstd2D as lt2D
#
# Define the type of lattice
grlt = lt2D.lattice2D()

print("Lattice J: %.1f and  Nd: %d " %(grlt.J_hop, grlt.Nd))

# Parameters
kx_int, kx_fin = -np.pi, np.pi
ky_int, ky_fin = -np.pi, np.pi
N_res = 100

kx_range = np.array([kx_int,kx_fin])
ky_range = np.array([ky_int,ky_fin])

kxR = np.linspace(kx_int, kx_fin,N_res)
kyR = np.linspace(ky_int, ky_fin,N_res)
#############

# to save quasienergies over k-space
E_arr = np.zeros((2,100,100), float)

# static Hamiltonian bandstructure calculation
E_arr = grlt.band_structure_static(kx_range, ky_range, N_res)
# dynamic quasienergies
# E_arr = grlt.band_structure_dynamic(kx_range, ky_range, N_res)


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

pl.show()

