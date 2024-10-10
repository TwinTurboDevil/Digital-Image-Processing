import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

def mean_filter(image, kernel_size):
    """
    Apply mean filtering to the input image.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (int): Size of the square kernel for mean filtering. Should be odd.

    Returns:
    numpy.ndarray: Mean-filtered image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create the mean filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Apply the filter to the image
    filtered_image = convolve(image, kernel, mode='reflect')

    return filtered_image

# Load an image using PIL
image_path = 'purple.jpg'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Check if the image has 3 channels (RGB)
if image_array.ndim == 3 and image_array.shape[2] == 3:
    # Apply mean filter to each channel separately
    filtered_image_array = np.zeros_like(image_array)
    for i in range(3):
        filtered_image_array[:, :, i] = mean_filter(image_array[:, :, i], kernel_size=5)
else:
    raise ValueError("The input image is not an RGB image.")

# Convert the result back to an image
filtered_image = Image.fromarray(np.uint8(filtered_image_array))

# Save the result
output_path = 'mean_filtered_purple.jpg'
filtered_image.save(output_path)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image Uisng Mean Filtering')
plt.imshow(filtered_image_array)
plt.axis('off')

plt.show()
