"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def calculate_nearest_neighbors(data, k):
    """
    Calculate k-nearest neighbors for each data point using Euclidean distance.

    Arguments:
    - data: a set of points of shape N x D, where N is the number of points and D is the dimension.
    - k: number of nearest neighbors to consider.

    Returns:
    - neighbors: a list of lists where each sublist contains the indices of the k nearest neighbors for each point.
    """
    num_points = data.shape[0]
    neighbors = []
    
    for i in range(num_points):
        # Calculate distances from point i to all other points
        distances = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
        
        # Get the indices of the k smallest distances
        # We use 'argsort' and exclude the first index since it will be the point itself
        nearest_neighbors = np.argsort(distances)[1:k+1]
        
        neighbors.append(nearest_neighbors)
        
    return np.array(neighbors)

def build_snn_graph(neighbors, smin):
    """
    Build a shared nearest neighbor graph.

    Arguments:
    - neighbors: array of nearest neighbors for each point.
    - smin: the minimum number of shared neighbors.

    Returns:
    - snn_graph: an adjacency matrix representing the SNN graph.
    """
    num_points = neighbors.shape[0]
    snn_graph = np.zeros((num_points, num_points), dtype=bool)
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # Compute the number of shared neighbors between points i and j
            shared_neighbors = np.intersect1d(neighbors[i], neighbors[j], assume_unique=True)
            if len(shared_neighbors) >= smin:
                snn_graph[i, j] = snn_graph[j, i] = True
                
    return snn_graph


# Identify clusters from the SNN graph
def identify_clusters(snn_graph):
    """
    Identify clusters from the SNN graph using a simple connected components algorithm.
    """
    # Find connected components
    num_points = snn_graph.shape[0]
    labels = -np.ones(num_points, dtype=np.int32)  # Unlabeled points are marked with -1
    cluster_id = 0
    
    for i in range(num_points):
        if labels[i] == -1:  # If the point is not yet labeled
            # Start a new cluster
            labels[i] = cluster_id
            # Find all points connected to point i
            points_to_visit = {i}
            while points_to_visit:
                current = points_to_visit.pop()
                # Get all neighbors connected to 'current' and not yet visited
                neighbors = np.where(snn_graph[current])[0]
                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        points_to_visit.add(neighbor)
            # Move to the next cluster ID
            cluster_id += 1
            
    return labels

# Calculate SSE and ARI
def calculate_sse_ari(data, labels, true_labels):
    """
    Calculate SSE and ARI given the cluster labels and true labels.
    """
    # Calculate SSE
    unique_labels = np.unique(labels)
    sse = 0
    for label in unique_labels:
        # Get the cluster points
        cluster_points = data[labels == label]
        # Compute the centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)
        # Sum of squared distances of points to the cluster centroid
        sse += np.sum((cluster_points - centroid) ** 2)
    
    # Calculate ARI
    # Create a contingency table
    true_labels = np.asarray(true_labels)
    labels = np.asarray(labels)
    n_classes = np.unique(true_labels).size
    n_clusters = np.unique(labels).size
    
    # Check if labels are non-negative and less then MAX_INT
    if (true_labels >= 0).all() and (labels >= 0).all():
        contingency = np.histogram2d(true_labels, labels, bins=(n_classes, n_clusters))[0]
    else:
        raise ValueError("Negative labels are not supported.")

    # Calculate the sums for rows, columns and the grand total
    sum_comb_c = sum(comb(n_c, 2) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in contingency.sum(axis=0))
    
    # Calculate the sum of the product of combinations for each cell in the contingency table
    prod_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())
    
    # Calculate the expected index (as if the dataset was randomly labeled)
    expected_index = sum_comb_c * sum_comb_k / comb(contingency.sum(), 2)
    
    # Calculate the observed index and the max index
    observed_index = prod_comb
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    # Calculate the adjusted Rand index
    ari = (observed_index - expected_index) / (max_index - expected_index)
    
    return sse, ari




def adjusted_rand_score(labels_true, labels_pred):
    # Create a contingency table
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n_classes = np.unique(labels_true).size
    n_clusters = np.unique(labels_pred).size
    
    # Check if labels are non-negative and less then MAX_INT
    if (labels_true >= 0).all() and (labels_pred >= 0).all():
        contingency = np.histogram2d(labels_true, labels_pred, bins=(n_classes, n_clusters))[0]
    else:
        raise ValueError("Negative labels are not supported.")

    # Calculate the sums for rows, columns and the grand total
    sum_comb_c = sum(comb(n_c, 2) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in contingency.sum(axis=0))
    
    # Calculate the sum of the product of combinations for each cell in the contingency table
    prod_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())
    
    # Calculate the expected index (as if the dataset was randomly labeled)
    expected_index = sum_comb_c * sum_comb_k / comb(contingency.sum(), 2)
    
    # Calculate the observed index and the max index
    observed_index = prod_comb
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    # Calculate the adjusted Rand index
    ari = (observed_index - expected_index) / (max_index - expected_index)
    
    return ari

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 5,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    
    # Calculate the nearest neighbors using the helper function
    neighbors = calculate_nearest_neighbors(data, params_dict['k'])

    # Build the SNN graph using the helper function
    snn_graph = build_snn_graph(neighbors, params_dict['smin'])

    # Identify clusters from the SNN graph using the helper function
    computed_labels = identify_clusters(snn_graph)

    # Calculate SSE and ARI using the helper function
    SSE, ARI = calculate_sse_ari(data, computed_labels, labels)

    # computed_labels: NDArray[np.int32] | None = None
    # SSE: float | None = None
    # ARI: float | None = None

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick
    
    # Load the data here
    data = np.load('question1_cluster_data.npy')
    true_labels = np.load('question1_cluster_labels.npy')


    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    # Parameter study configurations
    k_values = range(3, 9)
    smin_values = range(4, 11)
    groups = {}

    # Store SSE and ARI values for plotting
    sse_values = []
    ari_values = []

    for k in k_values:
        for smin in smin_values:
            # Assign parameters
            params_dict = {'k': k, 'smin': smin}
            # Apply the jarvis_patrick function to the first 10,000 data points
            computed_labels, SSE, ARI = jarvis_patrick(data[:10000], true_labels[:10000], params_dict)
            groups[(k, smin)] = {'ARI': ARI, 'SSE': SSE, 'labels': computed_labels}
            sse_values.append(SSE)
            ari_values.append(ARI)
            
            
    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    
    # Find the parameter sets with the largest ARI and smallest SSE
    max_ari_params = max(groups, key=lambda x: groups[x]['ARI'])
    min_sse_params = min(groups, key=lambda x: groups[x]['SSE'])

    # Record the SSE for the first group
    first_group_params = next(iter(groups))
    answers["1st group, SSE"] = groups[first_group_params]['SSE']
    

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = None
    plot_SSE = None
    
    # Generate scatter plot for the cluster with the largest ARI
    largest_ari_labels = groups[max_ari_params]['labels']
    plt.figure()
    plt.scatter(data[:10000, 0], data[:10000, 1], c=largest_ari_labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Scatter Plot with Largest ARI')
    plt.grid(True)
    plt.show()
    plot_ARI = plt
    
    # Generate scatter plot for the cluster with the smallest SSE
    smallest_sse_labels = groups[min_sse_params]['labels']
    plt.figure()
    plt.scatter(data[:10000, 0], data[:10000, 1], c=smallest_sse_labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Scatter Plot with Smallest SSE')
    plt.grid(True)
    plt.show()
    plot_SSE = plt

    
    
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster w scatterplotith smallest SSE"] = plot_SSE
    

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = np.mean(ari_values)

    # A single float
    answers["std_ARIs"] = np.std(ari_values)

    # A single float
    answers["mean_SSEs"] = np.mean(sse_values)

    # A single float
    answers["std_SSEs"] = np.std(sse_values)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
