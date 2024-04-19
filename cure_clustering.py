from sklearn.metrics import adjusted_rand_score
from datasets import nested_circles, interlocking_moons, blob_plus_sparse_outliers
from cure_dataset import (
    generate_cure_data,
    generate_well_separated_data,
    generate_varying_density_data,
    generate_non_globular_data,
)
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import NewType
from sklearn.metrics import silhouette_samples, silhouette_score

Data4 = tuple[
    NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]
]
Data2 = tuple[NDArray[np.float32], NDArray[np.int32]]


def cure_clustering(
    cluster_data: Data2, k: int, num_representatives: int, shrink_factor: float
) -> NDArray[np.int32]:
    """
    Perform CURE clustering on the given data.

    Parameters:
    - data: NDArray
        The input data to be clustered.
    - k: int
        The number of clusters to form.
    - num_representatives: int
        The number of representatives to select for each cluster.
    - shrink_factor: float
        The shrink factor used to update the representatives.

    Returns:
    - cluster_assignments: NDArray
        The cluster assignments for each data point in the input data.
    """

    data, labels = cluster_data[0], cluster_data[1]

    # Step 3: Hierarchical clustering on a random sample (10% of samples)
    # Sample 10% of the indices
    sample_indices = np.random.choice(
        data.shape[0], size=int(data.shape[0] * 0.1), replace=False
    )
    sample_data = data[sample_indices]
    # print(f"{sample_data.shape=}")  # 30 x 2

    nb_samples = 5
    # Generate the clusters from the `data` variable
    # clusters = {i: np.where(data[:, 2] == i)[0] for i in range(k)}
    # print("clusters= ", clusters)

    # Use Agglomerative Clustering to form initial clusters on sampled data
    hc = AgglomerativeClustering(n_clusters=k, linkage="complete")
    sample_labels = hc.fit_predict(sample_data)

    # Initialize clusters and representatives
    clusters = {i: data[sample_indices[sample_labels == i]] for i in range(k)}
    representatives = {}

    # Step 4: Initialize representatives per cluster
    for i in clusters:
        distances = cdist(
            clusters[i], [np.mean(clusters[i], axis=0)], metric="euclidean"
        )
        farthest_indices = np.argsort(distances.ravel())[-num_representatives:]
        representatives[i] = clusters[i][farthest_indices]
        representatives[i] = (1 - shrink_factor) * representatives[
            i
        ] + shrink_factor * np.mean(clusters[i], axis=0)

    # Step 5: Iterative re-assignment and update of representatives
    for _ in range(80):  # Assume a fixed number of iterations for convergence
        cluster_assignments = (
            np.argmin(
                cdist(
                    data,
                    np.vstack([representatives[i] for i in representatives]),
                    metric="euclidean",
                ),
                axis=1,
            )
            % k
        )

        # Update clusters based on new assignments
        for i in range(k):
            clusters[i] = data[cluster_assignments == i]
            if len(clusters[i]) > num_representatives:
                distances = cdist(clusters[i], [np.mean(clusters[i], axis=0)])
                farthest_indices = np.argsort(distances.ravel())[-num_representatives:]
                representatives[i] = clusters[i][farthest_indices]
                representatives[i] = (1 - shrink_factor) * representatives[
                    i
                ] + shrink_factor * np.mean(clusters[i], axis=0)

    # Representatives: dictionary: key=cluster index, value=representative points (x,y)
    cluster_assignments = assign_points_to_clusters(data, representatives) + 1
    labels = np.squeeze(labels)
    correct = np.sum(cluster_assignments == labels)
    # The formulas for error and acucracy is wrong since there is no way of knowing the label correspondances.
    # What is a better way to compute accuracy?
    # Compute the Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(labels, cluster_assignments)
    print("Adjusted Rand Index (ARI) = ", ari)

    # Compute the silhouette scores for each sample
    # silhouette_values = silhouette_samples(data[0], cluster_assignments)

    # # Compute the mean silhouette coefficient
    # mean_silhouette_coefficient = silhouette_score(data[0], cluster_assignments)

    # print("Silhouette sample values: ", silhouette_values)
    # print("Mean Silhouette Coefficient: ", mean_silhouette_coefficient)

    errors = np.sum(cluster_assignments != labels)
    accuracy = correct / (correct + errors)
    print("accuracy= ", accuracy)
    return cluster_assignments


# ----------------------------------------------------------------------
def assign_points_to_clusters(data, representatives):
    # This function will assign each point in 'data' to the cluster of its nearest representative.
    # 'representatives' is a dictionary where the key is the cluster index and the value is an array of representative points for that cluster.

    # Flatten the list of representatives and keep track of their cluster labels
    all_representatives = np.vstack([representatives[i] for i in representatives])
    representative_labels = np.concatenate(
        [[i] * len(representatives[i]) for i in representatives]
    )

    # Calculate the distance from each point to each representative
    distances = cdist(data, all_representatives)

    # Find the index of the nearest representative for each point
    nearest_representative_indices = np.argmin(distances, axis=1)

    # Map indices to cluster labels
    point_labels = representative_labels[nearest_representative_indices]

    return point_labels


# ----------------------------------------------------------------------

# Assuming 'data' is your dataset and 'final_representatives' is your final set of representatives
# obtained from the CURE algorithm after convergence:
# final_labels = assign_points_to_clusters(data, final_representatives)

# Now 'final_labels' contains the cluster index for each point in 'data'


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    k = 2
    num_representatives = 10
    shrink_factor = 0.4

    # Run CURE clustering
    # data = generate_cure_data()
    # data = generate_well_separated_data()
    # data = generate_non_globular_data()
    # data = generate_cure_data()
    # data = generate_varying_density_data()  # best results
    # data = nested_circles(
    # n_inner=300, n_outer=500, r_inner=1.0, r_outer=2.0, w_inner=0.1, w_outer=0.1
    # )
    # data = interlocking_moons(radius=1, n_points1=100, n_points2=100, offset=(0.7, 0.5))
    data = interlocking_moons(radius=1, n_points1=100, n_points2=100, offset=(0.3, 0.4))
    # data = blob_plus_sparse_outliers(n_dim=20, n_blob=800, n_outliers=200)
    print(f"Data len: {len(data)=}")

    labels = cure_clustering(data, k, num_representatives, shrink_factor)

    clusters = data[0]

    # Plot results
    plt.scatter(clusters[:, 0], clusters[:, 1], c=labels, alpha=0.5, cmap="viridis")
    plt.title("CURE Clustering Results")
    plt.show()

# from cure_dataset import generate_cure_data, generate_well_separated_data, generate_varying_density_data, generate_non_globular_data
