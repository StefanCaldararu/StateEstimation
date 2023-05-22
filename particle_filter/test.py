import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import wasserstein_distance

def generate_normal_distribution():
    """
    Generates a list of 100 points that fit perfectly under a normal distribution
    with mean 0 and standard deviation 0.8 and plots them.
    
    Returns:
        list: A list of 100 points that fit perfectly under the normal distribution.
    """
    # Create an array of 100 equally spaced points between -4 and 4
    x = np.linspace(-4, 4, 100)
    # Set the mean and standard deviation
    mu = 0
    sigma = 0.8
    
    # Generate a list of 100 points sampled from the normal distribution
    points = [random.gauss(mu, sigma) for _ in range(1000)]
    
    counts, bins = np.histogram(points, bins = 30)

    plt.stairs(counts, bins)

    # Plot the x vs y graph
    #plt.plot(x, points)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Normal Distribution with Mean 0 and Standard Deviation 0.8')
    plt.show()
    
    # Convert the numpy array to a list and return it
    points = y.tolist()
    
    return points


generate_normal_distribution()