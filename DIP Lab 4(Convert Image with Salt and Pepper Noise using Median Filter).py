import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter

def add_salt_and_pepper_noise(image, noise_level):
    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # Calculate the number of salt and pepper pixels
    num_salt = np.ceil(noise_level * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    
    # Add salt noise (white)
    noisy_image[coords[0], coords[1]] = 255
    
    # Add pepper noise (black)
    num_pepper = np.ceil(noise_level * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def apply_median_filter(image_array, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    filtered_image_array = np.zeros_like(image_array)

    for i in range(3):  # Assuming image_array has 3 channels (RGB)
        filtered_image_array[:, :, i] = median_filter(image_array[:, :, i], size=kernel_size)
    
    return filtered_image_array

# Load an image using PIL
image_path = 'purple.jpg'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Ensure the image is RGB
if image_array.ndim == 3 and image_array.shape[2] == 3:
    # Add 25% salt and pepper noise to the image
    noise_level = 0.25  # 25% noise
    noisy_image_array = add_salt_and_pepper_noise(image_array, noise_level)

    # Apply median filter
    kernel_size = 5  # You can adjust the kernel size to another odd number
    filtered_image_array = apply_median_filter(noisy_image_array, kernel_size)
else:
    raise ValueError("The input image is not an RGB image.")

# Convert the result back to an image
noisy_image = Image.fromarray(np.uint8(noisy_image_array))
filtered_image = Image.fromarray(np.uint8(filtered_image_array))

# Save the results
output_path_noisy = 'noisy_purple.jpg'
noisy_image.save(output_path_noisy)

output_path_filtered = 'median_filtered_noisy_purple.jpg'
filtered_image.save(output_path_filtered)

# Display the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image with Salt and Pepper Noise (25%)')
plt.imshow(noisy_image_array)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Filtered Image Using Median Filtering')
plt.imshow(filtered_image_array)
plt.axis('off')

plt.show()
