import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def fourier_time_series_prediction(y, num_frequencies=1000, steps=10):
    """
    Perform time series forward prediction using Fourier transform.

    Args:
        y (ndarray): Dependent variable of the time series.
        num_frequencies (int): Number of main frequencies to select. Default is 1000.
        steps (int): Number of steps for forward prediction. Default is 10.

    Returns:
        ndarray: Predicted results.
    """
    n = len(y)
    timestep = 1  # Assuming a unit time step of 1
    frequencies = np.fft.fftfreq(n, d=timestep)  # Compute the frequencies
    y_fft = fft(y)

    # Compute the amplitudes
    amplitudes = np.abs(y_fft) / n

    # Select the main frequency components
    top_indices = np.argsort(amplitudes)[::-1][:num_frequencies]
    top_frequencies = frequencies[top_indices]
    top_amplitudes = amplitudes[top_indices]

    # Perform inverse Fourier transform and forward prediction
    reconstructed = np.zeros(n + steps)
    for i in range(num_frequencies):
        reconstructed += np.cos(2 * np.pi * top_frequencies[i] * np.arange(n + steps)) * top_amplitudes[i]

    return reconstructed[n:]

# Sample data
y = data

# Perform forward prediction
steps = 100
predicted = fourier_time_series_prediction(y, steps=steps)

# Plot the original data and the predicted results
plt.plot(predicted, label='Original Data')
plt.plot(data[2001:2101],'b')  # Assuming `diff_states` is the original data
plt.legend()
plt.show()
