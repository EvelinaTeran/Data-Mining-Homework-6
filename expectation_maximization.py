import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def em_algorithm(data: NDArray[np.floating], max_iter: int = 100) -> tuple[
    tuple[np.floating] | None,
    tuple[np.floating] | None,
    tuple[NDArray[np.floating]] | None,
    tuple[NDArray[np.floating]] | None,
]:
    """
    Arguments:
    - data: numpy array of shape 50,000 x 2
    - max_iter: maximum number of iterations for the algorithm

    Return:
    - weights: the two coefficients \rho_1 and \rho_2 of the mixture model
    - means: the means of the two Gaussians (two scalars) as a list
    - covariances: the covariance matrices of the two Gaussians
      (each is a 2x2 symmetric matrix) return the full matrix
    - log_likelihoods: `max_iter` values of the log_likelihood, including the initial value

    Notes:
    - order the distribution parameters such that the x-component of
          the means are ordered from largest to smallest.
    - hint: the log-likelihood is monotonically increasing (or constant)
          if the algorithm is implemented correctly.
    - If this code is copied from some source, make sure to reference the
        source in this doc-string.
    """
    # CODE FILLED BY STUDENT

    weights = None
    means = None
    covariances = None
    log_likelihoods = None

    return weights, means, covariances, log_likelihoods


# ----------------------------------------------------------------------
def gaussian_mixture():
    """
    Calculate the parameters of a Gaussian mixture model using the EM algorithm.
    Specialized to two distributions.
    """
    answers = {}

    # ADD STUDENT CODE

    # Return a 2D scatterplot of the clusters, colored by id
    # plot_cluster = plt.scatter(....)
    answers["plot_original_cluster"] = plot_cluster

    # Return the `em_algorithm` funtion 
    answers["em_algorithm_function"] = em_algorithm

    # 1D numpy array of floats
    answers["log_likelihood"] = None

    # a line plot using matplotlib.pyplot.plot
    # Make sure to include title, axis labels, and a grid.
    # Save the plot to file "plot_log_likelihood.pdf", and add to your report.
    answers["plot_log_likelihood"] = None

    # list with the mean and standard deviation (over 10 trials) of the mean vector
    # of the first distribution
    answers["probability_1_mean"] = None
    answers["probability_2_mean"] = None

    # list with the mean and standard deviation (over 10 trials) of the covariance matrix
    # of the first distribution. The covariance matrix should be in the standard order.
    # (https://www.cuemath.com/algebra/covariance-matrix/)
    # 
    answers["probability_1_covariance"] = None
    answers["probability_2_covariance"] = None

    # list with the mean and standard deviation (over 10 trials) of the amplitude \rho_1
    # of the first distribution.
    answers["probability_1_amplitude"] = None
    answers["probability_2_amplitude"] = None

    # Return a 2x2 numpy Array of floats. Average of the confusion matrices of the 10 trials
    answers["average_confusion_matrix"] = None

    # Return a 2x2 numpy Array of floats. Standard deviation of the confusion matrices of the 10 trials
    # This means to take the standard deviation of each element of the confusion matrix. 
    # So there are 10 (1,1) elements, and you are to average these and take the standard deviation. 
    answers["std_confusion_matrix"] = None

    # Return a list of 10 ARIs (float), one for each sample of 5,000 points
    answers["ARI"] = None

    # Return a list of 10 SSEs (float), one for each sample of 5,000 points
    answers["SSE"] = None


    # Return the mean and standard deviation of the 10 trials (of 5000 points each)
    answers["avg_std_ARI"] = None

    # Return the mean and standard deviation of the 10 trials (of 5000 points each)
    answers["avg_std_SSE"] = None

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = gaussian_mixture()
    with open("gaussian_mixture.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
