import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import random

# Parameters
N = 50
STEP = (N**2) * 100
temp_step = 0.01
T_plus = 0.1
burn_value = 0.2
burn_in = int(STEP * burn_value)
T_range = np.arange(5, 0.0001, -temp_step)
spin = np.random.choice([1, -1], size=(N, N))

# Optimized Hamiltonian function
def optimized_Hamiltonian(i, j, J, spin, N):
    return J * spin[i, j] * (spin[(i - 1) % N, j] + spin[(i + 1) % N, j] + spin[i, (j - 1) % N] + spin[i, (j + 1) % N])

# Optimized Metropolis Algorithm
def optimized_Metropolis_Algorithm(i, j, T, J, spin, N):
    delta_H = 2 * optimized_Hamiltonian(i, j, J, spin, N)
    if delta_H < 0 or np.random.random() < np.exp(-delta_H / T):
        spin[i, j] *= -1

# Optimized Ising model
def optimized_Ising_model(N, T, J, STEP, spin):
    Magnetization_array = np.zeros(STEP - burn_in)

    # Batch generating random indices
    random_indices = np.random.randint(0, N, size=(STEP, 2))

    for s in range(STEP):
        i, j = random_indices[s]
        optimized_Metropolis_Algorithm(i, j, T, J, spin, N)
        if s >= burn_in:
            Magnetization_array[s - burn_in] = np.mean(spin)

    average_final_Magnetization = np.mean(Magnetization_array)
    return average_final_Magnetization , Magnetization_array

# Worker function for multiprocessing
def worker(T):
    local_spin = np.random.choice([1, -1], size=(N, N))  # Create a local spin array for each process
    avg_mag , _ = optimized_Ising_model(N, T, 1, STEP, local_spin)
    return avg_mag


# Parallel processing with timing
def Ising_model_T_parallel(T_range):
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(T_range)) as pbar:
            average_magnetizations = []
            for result in pool.imap(worker, T_range):
                average_magnetizations.append(result)
                pbar.update(1)
    return average_magnetizations

#autocorrelation
def autocorrelation(magnetization_array):
    n = len(magnetization_array)
    mean_mag = np.mean(magnetization_array)
    var_mag = np.var(magnetization_array)

    autocorr = np.correlate(magnetization_array - mean_mag, magnetization_array - mean_mag, mode='full') / (var_mag * n)
    return autocorr[n - 1:]


# Run the parallel processing function
average_magnetizations = Ising_model_T_parallel(T_range)

# Call function for a specific temperature to get magnetization vs. step
average_magnetization,magnetization_steps = optimized_Ising_model(N, T_plus, 1, STEP, spin.copy())
auto_corr = autocorrelation(magnetization_steps)

# Plotting results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Average Magnetization per Temperature
axes[0].scatter(T_range, average_magnetizations)
axes[0].set_xlabel("Temperature (T)")
axes[0].set_ylabel("Average Magnetization (M)")
axes[0].set_title(f'Average Magnetization per Temperature in {N}*{N} lattice')

# Plot Magnetization vs. Step for a specific temperature
axes[1].plot(np.arange(STEP - int(STEP * burn_value)), magnetization_steps)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Magnetization")
axes[1].set_title(f'Magnetization vs. Step at T = {T_plus}')



plt.tight_layout()
plt.show()


# Plotting the Autocorrelation
plt.figure(figsize=(8, 4))
plt.plot(auto_corr)
plt.xlabel("Time lag (τ)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of Magnetization")
plt.show()