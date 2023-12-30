import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

# Parameters
N = 40
STEP = (N ** 2) * 1000
temp_step = 0.01
T_plus = 2.23
burn_value = 0.2
burn_in = int(STEP * burn_value)
T_range = np.arange(5, 0.0001, -temp_step)
spin = np.random.choice([1, -1], size=(N, N))


# Optimized Hamiltonian function
def optimized_Hamiltonian(i, j, J, spin, N):
    return J * spin[i, j] * (spin[(i - 1) % N, j] + spin[(i + 1) % N, j]
                             + spin[i, (j - 1) % N] + spin[i, (j + 1) % N])


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
    return average_final_Magnetization, Magnetization_array


# Worker function for multiprocessing
def worker(T):
    local_spin = np.random.choice([1, -1], size=(N, N))
    avg_mag, _ = optimized_Ising_model(N, T, 1, STEP, local_spin)
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


# autocorrelation over step at specefic T
def autocorrelation(magnetization_array):
    n = len(magnetization_array)
    mean_mag = np.mean(magnetization_array)
    var_mag = np.var(magnetization_array)

    autocorr = np.correlate(magnetization_array - mean_mag,
                            magnetization_array - mean_mag,
                            mode='full') / (var_mag * n)
    return autocorr[n - 1:]


# # correlation over T
# def correlation(average_magnetization):
#     n = len(average_magnetizations)
#     mean_mag_T = np.mean(average_magnetization)
#     var_mag_T = np.var(average_magnetization)

#     corr = np.correlate(average_magnetization - mean_mag_T
#                         ,average_magnetization - mean_mag_T,
#                          mode='full') / (var_mag_T* n)
#     return corr[n - 1:]


# Run the parallel processing function
average_magnetizations = Ising_model_T_parallel(T_range)

# Call function for a specific temperature to get magnetization vs. step
average_magnetization, magnetization_steps = optimized_Ising_model(N, T_plus,
                                                                   1, STEP, spin.copy())
auto_corr = autocorrelation(magnetization_steps)
# corr = correlation(average_magnetization)

# Plotting results
fig, axes = plt.subplots(4, 1, figsize=(9, 12))

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

# Plotting the Autocorrelation
axes[2].plot(auto_corr)
axes[2].set_xlabel("Time lag (τ)")
axes[2].set_ylabel("Autocorrelation")
axes[2].set_title(f'Autocorrelation of Magnetization at T = {T_plus}')

# # Plotting the correlation
# axes[3].plot(corr)
# axes[3].set_xlabel(" Temp lag ")
# axes[3].set_ylabel("correlation")
# axes[3].set_title(f'correlation of Magnetization over T ')


plt.tight_layout()
plt.show()

plt.savefig(f'T={T_plus},STEP={STEP},temp_step={temp_step},N={N}.jpg')

