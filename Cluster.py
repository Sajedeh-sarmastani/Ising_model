from scipy.ndimage import measurements



def calculate_cluster_sizes(spin)
    labeled_array, num_features = measurements.label(spin)
    cluster_sizes = measurements.sum(spin, labeled_array, index=np.arange(num_features + 1))
    return cluster_sizes[cluster_sizes  0]

def max_cluster_size(cluster_sizes)
    return np.max(cluster_sizes)


average_sizes = []
lattice_size = []

for i in range(10)
    N = 25  (i + 1)
    spin = np.random.choice([1, -1], size=(N, N))
    spin = Ising_model(N, T=T_plus, J=1, step=N2  100)
    spin[spin == -1] = 0
    cluster_sizes = calculate_cluster_sizes(spin)
    avg_size = max_cluster_size(cluster_sizes)
    average_sizes.append(avg_size)
    lattice_size.append(N)



plt.plot(lattice_size, average_sizes, marker='o')
plt.xlabel('Lattice Size (N)')
plt.ylabel('Average Cluster Size')
plt.title('Dependency of max_Cluster Size on Lattice Size in Ising Model at critical temp')
plt.grid(True)
plt.show()
plt.savefig(Dependency of max_Cluster Size on Lattice Size )

