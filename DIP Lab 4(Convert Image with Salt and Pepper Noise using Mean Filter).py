import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

def mean_filter(image, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    filtered_image = convolve(image, kernel, mode='reflect')
    return filtered_image

def add_salt_and_pepper_noise(image, noise_level):
    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # Calculate number of salt and pepper pixels
    num_salt = np.ceil(noise_level * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    
    # Add salt noise
    noisy_image[coords[0], coords[1]] = 255  # Salt noise (white)
    
    # Add pepper noise
    num_pepper = np.ceil(noise_level * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # Pepper noise (black)
    
    return noisy_image

# Load the image
image_path = 'purple.jpg'  # Replace with your image path
image = Image.open(image_path)
image_array = np.array(image)

if image_array.ndim == 3 and image_array.shape[2] == 3:
    # Add salt and pepper noise to the image
    noise_level = 0.25  # 25% noise
    noisy_image_array = add_salt_and_pepper_noise(image_array, noise_level)

    # Apply mean filter to each channel separately
    filtered_image_array = np.zeros_like(noisy_image_array)
    for i in range(3):
        filtered_image_array[:, :, i] = mean_filter(noisy_image_array[:, :, i], kernel_size=5)
else:
    raise ValueError("The input image is not an RGB image.")

# Convert the result back to an image
filtered_image = Image.fromarray(np.uint8(filtered_image_array))
noisy_image = Image.fromarray(np.uint8(noisy_image_array))

# Save the results
output_path_noisy = 'noisy_purple.jpg'
noisy_image.save(output_path_noisy)

output_path_filtered = 'mean_filtered_noisy_purple.jpg'
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
plt.title('Filtered Image Using Mean Filtering')
plt.imshow(filtered_image_array)
plt.axis('off')

plt.show()
