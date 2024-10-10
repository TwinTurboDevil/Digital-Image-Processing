import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def apply_gaussian_filter(image_array, sigma):
    """
    Apply Gaussian filtering to each channel of the RGB image.

    Parameters:
    image_array (numpy.ndarray): Input RGB image.
    sigma (float): Standard deviation for Gaussian kernel. Controls the amount of smoothing.

    Returns:
    numpy.ndarray: Gaussian-filtered RGB image.
    """
    # Create an empty array to store the filtered image
    filtered_image_array = np.zeros_like(image_array)

    # Apply Gaussian filter to each channel separately
    for i in range(3):  # Assuming image_array has 3 channels (RGB)
        filtered_image_array[:, :, i] = gaussian_filter(image_array[:, :, i], sigma=sigma)
    
    return filtered_image_array

# Load an image using PIL
image_path = 'purple.jpg'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Ensure the image is RGB
if image_array.ndim == 3 and image_array.shape[2] == 3:
    # Apply Gaussian filter
    sigma = 1.0  # Standard deviation for Gaussian kernel. Adjust as needed.
    filtered_image_array = apply_gaussian_filter(image_array, sigma)
else:
    raise ValueError("The input image is not an RGB image.")

# Convert the result back to an image
filtered_image = Image.fromarray(np.uint8(filtered_image_array))

# Save the result
output_path = 'gaussian_filtered_purple.jpg'
filtered_image.save(output_path)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image Using Gaussian Filtering')
plt.imshow(filtered_image_array)
plt.axis('off')

plt.show()
