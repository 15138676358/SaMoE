"""
data_generator.py
This module provides a function to generate synthetic data for testing purposes.
The data generator is designed as follows:
1. randomize the mean and std of a normal distribution: (mu, sigma), all of which are in the range of (0, 1)
2. smaple 5 numbers from the uniform distribution: X \in (0, 1)
3. calculate the y values using the normal distribution probability density function (PDF):
   y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
4. divide the data into context, input and output_gt:
   - context: [X[0:4], y[0:4]]
   - input: [X[4]]
   - output_gt: [y[4]]
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm


def generate_synthetic_data():
    """
    Generate synthetic data for testing purposes.
    
    Returns:
        context (list): A list containing the first four X values and their corresponding y values.
        input (list): A list containing the fifth X value.
        output_gt (list): A list containing the ground truth y value for the fifth X.
    """
    # Randomize mu and sigma from a normal distribution
    mu = np.random.normal(0.65, 0.2)  # mean = 0.5, std = 0.2
    sigma = np.random.uniform(0.2, 0.8)

    # Sample 5 numbers from the uniform distribution
    X = np.random.uniform(0, 1, 5)

    # Calculate y values using the normal distribution PDF
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)

    # Divide the data into context, input and output_gt
    mu = np.array([mu])
    sigma = np.array([sigma])
    context = np.column_stack((X[:4], y[:4])).reshape(-1)
    input = np.array(X[4:])
    output_gt = np.array(y[4:])

    return mu, sigma, context, input, output_gt

def test_generate_synthetic_data():
    """
    Test the generate_synthetic_data function.
    This function is a placeholder for unit tests.
    """
    mu, sigma, context, input_data, output_gt = generate_synthetic_data()
    plt.scatter(context[:, 0], context[:, 1], label='Context')
    plt.scatter(input_data, output_gt, color='red', label='Input and Output GT')
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('mu: {:.2f}, sigma: {:.2f}'.format(mu, sigma))
    plt.legend()
    plt.show()

def generate_data_file(data_number=1000, file_name="test_data.npz"):
    """
    Generate synthetic data and save it to a file.
    This function is a placeholder for future implementation.
    """
    all_mu, all_sigma, all_context, all_input, all_output_gt = [], [], [], [], []
    for _ in tqdm.tqdm(range(data_number), desc="Generating synthetic data"):
        mu, sigma, context, input_data, output_gt = generate_synthetic_data()
        all_mu.append(mu)
        all_sigma.append(sigma)
        all_context.append(context)
        all_input.append(input_data)
        all_output_gt.append(output_gt)
    all_mu, all_sigma, all_context, all_input, all_output_gt = np.array(all_mu), np.array(all_sigma), np.array(all_context), np.array(all_input), np.array(all_output_gt)

    np.savez(file_name, mu=all_mu, sigma=all_sigma, context=all_context, input=all_input, output_gt=all_output_gt)

    print(f"Data saved to {file_name}")
 


if __name__ == "__main__":
    # test_generate_synthetic_data()
    generate_data_file()