import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fftshift, ifftshift

# Function to compute FFT of the image
def compute_fft(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)  # Convert to a NumPy array

    # Compute 2D FFT
    fft_image = np.fft.fft2(img_array)
    
    # Shift the zero frequency component to the center
    fft_image_shifted = fftshift(fft_image)
    
    # Compute magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_image_shifted) + 1)

    # Compute phase spectrum
    phase_spectrum = np.angle(fft_image_shifted)

    return img_array, magnitude_spectrum, phase_spectrum

# Path to your image file
image_path = 'purple.jpg'  # Replace with your image path

# Compute the FFT, magnitude spectrum, and phase spectrum
original_image, fft_magnitude_spectrum, fft_phase_spectrum = compute_fft(image_path)

# Plotting the original image, FFT magnitude spectrum, and FFT phase spectrum
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot original image
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot FFT magnitude spectrum
axes[1].imshow(fft_magnitude_spectrum, cmap='gray')
axes[1].set_title('FFT Magnitude Spectrum')
axes[1].axis('off')

# Plot FFT phase spectrum
# Since the phase spectrum contains values between -π and π, it's already in a range suitable for plotting.
axes[2].imshow(fft_phase_spectrum, cmap='gray')
axes[2].set_title('FFT Phase Spectrum')
axes[2].axis('off')

plt.show()
