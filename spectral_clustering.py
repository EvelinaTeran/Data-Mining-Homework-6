"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
import scipy.special
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################
def compute_affinity_matrix(X: NDArray[np.floating], sigma: float) -> NDArray[np.floating]:
    """
    Computes the affinity matrix using a Gaussian kernel.
    """
    # Compute the squared Euclidean distance matrix
    squared_dist_matrix = cdist(X, X, 'sqeuclidean')
    # Compute the affinity matrix using the Gaussian kernel
    affinity_matrix = np.exp(-squared_dist_matrix / (2 * sigma ** 2))
    return affinity_matrix


def compute_laplacian(W: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Computes the Laplacian of the affinity matrix.
    """
    # Compute the degree matrix
    D = np.diag(np.sum(W, axis=1))
    # Compute the Laplacian
    L = D - W
    return L


# def calculate_ari(labels_true: NDArray[np.int32], labels_pred: NDArray[np.int32]) -> float:
#     """
#     Calculate the Adjusted Rand Index (ARI) without using sklearn.
#     """
#     # Create the contingency table
#     classes, class_idx = np.unique(labels_true, return_inverse=True)
#     clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
#     n = len(labels_true)
#     contingency = np.histogram2d(class_idx, cluster_idx, bins=(len(classes), len(clusters)))[0]

#     # Sum over rows & columns
#     sum_comb_c = sum(scipy.special.comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
#     sum_comb_k = sum(scipy.special.comb(n_k, 2) for n_k in np.sum(contingency, axis=0))

#     # Sum over the whole matrix
#     sum_comb = sum(scipy.special.comb(n_ij, 2) for n_ij in contingency.flatten())

#     # Calculate the expected index (combining the sum_comb for classes and clusters)
#     prod_comb = sum_comb_c * sum_comb_k / scipy.special.comb(n, 2)
    
#     # Calculate the index (same pairs)
#     index = sum_comb - prod_comb
    
#     # Calculate the max index
#     max_index = (sum_comb_c + sum_comb_k) / 2
    
#     # Calculate the adjusted index
#     adjusted_index = index - prod_comb
    
#     # Calculate the ARI
#     ARI = adjusted_index / (max_index - prod_comb)
    
#     return ARI

def calculate_ari(labels_true: NDArray[np.int32], labels_pred: NDArray[np.int32]) -> float:
    """
    Calculate the Adjusted Rand Index (ARI) without using sklearn.
    """
    try:
        # Create the contingency table
        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        n = len(labels_true)
        if n == 0:
            print("Warning: No labels provided.")
            return 0.0
        
        contingency = np.histogram2d(class_idx, cluster_idx, bins=(len(classes), len(clusters)))[0]

        # Sum over rows & columns
        sum_comb_c = sum(scipy.special.comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
        sum_comb_k = sum(scipy.special.comb(n_k, 2) for n_k in np.sum(contingency, axis=0))

        # Sum over the whole matrix
        sum_comb = sum(scipy.special.comb(n_ij, 2) for n_ij in contingency.flatten())

        # Calculate the expected index (combining the sum_comb for classes and clusters)
        prod_comb = sum_comb_c * sum_comb_k / scipy.special.comb(n, 2)
        
        # Calculate the index (same pairs)
        index = sum_comb - prod_comb
        
        # Calculate the max index
        max_index = (sum_comb_c + sum_comb_k) / 2
        
        # Calculate the adjusted index
        adjusted_index = index - prod_comb
        
        # Calculate the ARI
        ARI = adjusted_index / (max_index - prod_comb)
        
        return ARI
    except Exception as e:
        print(f"Failed to calculate ARI: {e}")
        return 0.0  # Return a default value in case of any failure


# def custom_kmeans(features: NDArray[np.floating], num_clusters: int) -> tuple[NDArray[np.int32], NDArray[np.floating]]:
#     """
#     Perform k-means clustering using scipy and return the labels and cluster centers.
#     """
#     centroid, label = kmeans2(features, num_clusters, minit='random')
#     return label, centroid

def custom_kmeans(features: NDArray[np.floating], num_clusters: int, attempts=5) -> tuple[NDArray[np.int32], NDArray[np.floating]]:
    """
    Perform k-means clustering using scipy and return the labels and cluster centers.
    Attempts to resolve the issue if one of the clusters is empty.
    """
    for attempt in range(attempts):
        try:
            centroid, label = kmeans2(features, num_clusters, minit='++' if attempt > 0 else 'random')
            return label, centroid
        except ValueError:
            continue
    raise RuntimeError("k-means failed to converge after several attempts.")


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    
    # Extract the parameters
    sigma = params_dict['sigma']
    k = params_dict['k']

    try:
        # Step 1: Construct the affinity matrix
        A = compute_affinity_matrix(data, sigma)
    
        # Step 2: Compute the Laplacian matrix
        L = compute_laplacian(A)
    
        # Step 3: Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(L, k, which='SM')
    
        # Step 4: Use k-means on the eigenvectors to form clusters
        computed_labels, _ = custom_kmeans(eigenvectors, k)
    
        # Recalculate centroids in the original space
        centroids = np.array([data[computed_labels == i].mean(axis=0) for i in range(k)])
        
        print("Data shape:", data.shape)
        print("Centroids shape:", centroids.shape)
        print("Filtered data shape for a cluster:", data[computed_labels == 0].shape)
        
        # Step 5: Compute the SSE and ARI
        SSE = np.sum([np.sum((data[computed_labels == i] - centroids[i])**2) for i in range(k)])
        ARI = calculate_ari(labels, computed_labels)
        
    
        computed_labels: NDArray[np.int32] | None = None
        SSE: float | None = None
        ARI: float | None = None
        eigenvalues: NDArray[np.floating] | None = None
        
    
        return computed_labels, SSE, ARI, eigenvalues
    except Exception as e:
        print(f"Error processing sigma={sigma}: {e}")
        return None, np.inf, 0.0, None  # Default values in case of failure


def spectral_clustering():
    """
    Performs SPECTRAL clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    # Load the data
    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    answers = {
        "spectral_function": None,  
        "cluster parameters": {},
        "1st group, SSE": None,
        "cluster scatterplot with largest ARI": None,
        "cluster scatterplot with smallest SSE": None,
        "eigenvalue plot": None,
        "mean_ARIs": None,
        "std_ARIs": None,
        "mean_SSEs": None,
        "std_SSEs": None,
    }

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    sigma_range = np.linspace(0.1, 10, 10)
    groups = {}
    best_ARI = float('-inf')
    best_SSE = float('inf')
    best_params = {'ARI': {}, 'SSE': {}}
    ARIs = []
    SSEs = []
    eigenvalues_list = []

    # Take 5 random slices of 1000 points each
    random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
    sliced_data = data[random_indices]
    sliced_labels = labels[random_indices]

    data_groups = np.array_split(sliced_data, 5)
    label_groups = np.array_split(sliced_labels, 5)

    # for i, (data_group, label_group) in enumerate(zip(data_groups, label_groups)):
    #     for sigma in sigma_range:
    #         params_dict = {'sigma': sigma, 'k': 5}
    #         computed_labels, SSE, ARI, eigenvalues = spectral(data_group, label_group, params_dict)

    #         groups[i] = {'sigma': sigma, 'ARI': ARI, 'SSE': SSE}
    #         ARIs.append(ARI)
    #         SSEs.append(SSE)
    #         eigenvalues_list.append(eigenvalues)

    #         if ARI > best_ARI:
    #             best_ARI = ARI
    #             best_params['ARI'] = params_dict

    #         if SSE < best_SSE:
    #             best_SSE = SSE
    #             best_params['SSE'] = params_dict
            
    #         # Store SSE for the first group separately
    #         if i == 0:  # This is the first group
    #             answers["1st group, SSE"] = SSE
    
    for i, (data_group, label_group) in enumerate(zip(data_groups, label_groups)):
        for sigma in sigma_range:
            params_dict = {'sigma': sigma, 'k': 5}
            computed_labels, SSE, ARI, eigenvalues = spectral(data_group, label_group, params_dict)

            if ARI is not None and ARI > best_ARI:
                best_ARI = ARI
                best_params['ARI'] = params_dict

            if SSE is not None and SSE < best_SSE:
                best_SSE = SSE
                best_params['SSE'] = params_dict

            groups[i] = {'sigma': sigma, 'ARI': ARI, 'SSE': SSE}
            ARIs.append(ARI)
            SSEs.append(SSE)
            eigenvalues_list.append(eigenvalues)

            if i == 0:  # First group special handling
                answers["1st group, SSE"] = SSE

    answers["cluster parameters"] = groups
    
    ARIs = [ari for ari in ARIs if ari is not None]  # Ensure no None values
    SSEs = [sse for sse in SSEs if sse is not None]  # Ensure no None values
    
    if ARIs:  # Ensure ARIs is not empty to avoid ValueError in np.mean
        answers["mean_ARIs"] = np.mean(ARIs)
        answers["std_ARIs"] = np.std(ARIs)
    else:
        answers["mean_ARIs"] = None
        answers["std_ARIs"] = None

    if SSEs:  # Ensure SSEs is not empty to avoid ValueError in np.mean
        answers["mean_SSEs"] = np.mean(SSEs)
        answers["std_SSEs"] = np.std(SSEs)
    else:
        answers["mean_SSEs"] = None
        answers["std_SSEs"] = None
        
    # answers["mean_ARIs"] = np.mean(ARIs)
    # answers["std_ARIs"] = np.std(ARIs)
    # answers["mean_SSEs"] = np.mean(SSEs)
    # answers["std_SSEs"] = np.std(SSEs)

    # Visualizations
    # Filtering out failures before plotting
    valid_sse_indices = [i for i, sse in enumerate(SSEs) if sse != np.inf]
    valid_ari_indices = [i for i, ari in enumerate(ARIs) if ari != 0.0]

    
    # Plot SSE and ARI for different sigma values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(sigma_range, SSEs, c=SSEs, cmap='viridis')
    plt.colorbar(label='SSE')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.title('SSE by Sigma')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(sigma_range, ARIs, c=ARIs, cmap='viridis')
    plt.colorbar(label='ARI')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.title('ARI by Sigma')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot of the eigenvalues for the best parameter based on ARI
    plt.figure()
    plt.plot(np.sort(eigenvalues_list[np.argmax(ARIs)]), marker='o')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues for Best ARI')
    plt.grid(True)
    plt.show()
    
    # Store plots in answers dictionary
    answers["cluster scatterplot with largest ARI"] = plt.scatter(data[:, 0], data[:, 1], c=computed_labels, cmap='viridis')
    answers["cluster scatterplot with smallest SSE"] = plt.scatter(data[:, 0], data[:, 1], c=computed_labels, cmap='viridis')
    answers["eigenvalue plot"] = plt.plot(np.sort(eigenvalues_list[np.argmax(ARIs)]), marker='o')
    
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)