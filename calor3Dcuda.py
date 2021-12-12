#------------------------------------------------
#  Solución numérica a la ecuación de difusión 
#  en 3D usando diferencias finitas explícitas
#------------------------------------------------
#  Escrito en python
#  Acelerado con numba-cuda
#------------------------------------------------
#  Autor: Dr. Julián T. Becerra Sagredo
#  ESFM IPN
#------------------------------------------------

import numpy as np 
from mayavi import mlab
import time
from numba import jit
from numba import cuda
from numba import *

@jit
def evolucion(u,n_0,n_1,n_2,udx2_0,udx2_1,udx2_2,dt,kd,i):
    jp1 = i + n_0
    jm1 = i - n_0
    kp1 = i + n_0*n_1
    km1 = i - n_0*n_1
    laplaciano = (u[i-1]-2.0*u[i]+u[i+1])*udx2_0 + \
                 (u[jm1]-2.0*u[i]+u[jp1])*udx2_1 + \
                 (u[km1]-2.0*u[i]+u[kp1])*udx2_2
    unueva = u[i] + dt*kd*laplaciano
    return unueva

evolucion_gpu = cuda.jit(device=True)(evolucion)

@cuda.jit
def solucion_kernel(u_d,un_d,udx2_0,udx2_1,udx2_2,dt,n_0,n_1,n_2,kd):
   ii, jj , kk = cuda.grid(3)
   i = ii + n_0*jj + n_0*n_1*kk
   if ii==0 or jj==0 or kk==0 or ii==n_0-1 or jj==n_1-1 or kk==n_2-1: 
      unueva = 0.0
   else: 
      unueva = evolucion_gpu(u_d,n_0,n_1,n_2,udx2_0,udx2_1,udx2_2,dt,kd,i)
   if i == int((n_0*n_1*n_2)/2)+int(n_0*n_1/2)+int(n_0/2):
      unueva = 1.0
   un_d[i] = unueva

#---------------------------
#  PROGRAMA PRINCIPAL
#---------------------------
if __name__ == "__main__":
   #-------------------------------------------
   # Número de celdas
   n = np.array([32,32,32],dtype=np.int64)
   # Tamaño del dominio (menor que uno)
   L = np.array([1.0,1.0,1.0],dtype=np.float64)
   # Constante de difusión
   kd:float64 = 0.2
   # Pasos de tiempo
   pasos:int = 100000
   blockdim = (8,4,4)                                      
   #-------------------------------------------

   # Bloques
   griddim = (int(n[0]/blockdim[0]),int(n[1]/blockdim[1]),int(n[2]/blockdim[2])) 
   # Tamaño de las celdas
   dx = L/n
   udx2 = 1.0/(dx*dx)
   # Paso de tiempo
   dt = 0.1*(min(dx[0],dx[1],dx[2])**2)/kd
   print("dt = ",dt)
   # Total de celdas
   nt = n[0]*n[1]*n[2]
   print("celdas = ",nt)
   start = time.time()
   # Llenar la solución con ceros
   u  = np.zeros(nt,dtype=np.float64)  # arreglo de lectura
   un = np.zeros(nt,dtype=np.float64)  # arreglo de escritura
   # Pasar arreglos al GPU
   u_d = cuda.to_device(u) 
   un_d = cuda.to_device(un)
   # Integrar en el tiempo
   for t in range(1,pasos+1):
     solucion_kernel[griddim,blockdim](u_d,un_d,udx2[0],udx2[1],udx2[2],dt,n[0],n[1],n[2],kd)
     u_d = cuda.to_device(un_d)
     if t%100==0: print("paso = ",t)
   # Pasar arreglo al CPU
   u_d.copy_to_host(u)
   end = time.time()
   print("Tardó: ",end-start,"s")
   #---------------------------------------
   #-----------------
   # Graficar en 3D
   #-----------------
   u = np.reshape(u,(n[0],n[1],n[2]))
   x,y,z = np.ogrid[0:L[0]:1j*n[0],0:L[1]:1j*n[1],0:L[2]:1j*n[2]]
   src = mlab.pipeline.scalar_field(u)
   mlab.pipeline.iso_surface(src, contours=[u.min()+0.1*u.ptp(), ],opacity=0.3)
   mlab.pipeline.iso_surface(src, contours=[u.max()-0.1*u.ptp(), ],)
   mlab.show()
