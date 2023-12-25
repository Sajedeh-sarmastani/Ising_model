
# visulizing dependency of cluster to lattice size


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import measurements




temp_step = 0.01
T_plus = 2.269185


def Ising_model(N, T, J ,step):


    def Hamiltonian(i,j):
        return J * spin[i][j] * (spin[(i - 1) % N][j] + spin[(i + 1) % N][j] + spin[i][(j - 1) % N] + spin[i][(j + 1) % N])

    def Metropolice_Algoritm(delta_H):
        if delta_H < 0 or np.random.random() < np.exp(- delta_H / T):
            spin [i][j] *= -1
            return spin

    #array for  sum over magnetization over all spins per step
    Magnetization_array = np.zeros(step)

    for s in range(step):
        i ,j = np.random.randint(0, N ,2)
        delta_H = 2 * Hamiltonian(i,j)
        Metropolice_Algoritm(delta_H)
        
    return spin



    # np.save(f"results/{i}.npy", spin)

fig, axes = plt.subplots(10, 2, figsize=(15, 50)) 
for i in range(10):  
    N = 100 * (i + 1)  
    spin = np.random.choice([1, -1], size=(N, N))
    spin = Ising_model(N, T=T_plus, J=1, step=N**2 * 100)
    spin[spin == -1] = 0

    im0 = axes[i][0].imshow(spin, origin='lower', interpolation='nearest') 
    axes[i][0].title.set_text("Matrix")  
    fig.colorbar(im0, ax=axes[i][0])  

    lw, num = measurements.label(spin)
    area = measurements.sum(spin, lw, index=np.arange(lw.max() + 1))
    areaImg = area[lw]
    im1 = axes[i][1].imshow(areaImg, origin='lower', interpolation='nearest')  
    axes[i][1].title.set_text("Clusters by area")  
    fig.colorbar(im1, ax=axes[i][1]) 
plt.tight_layout()
plt.show()
      
