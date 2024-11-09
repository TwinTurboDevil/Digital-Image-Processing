from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from scipy.ndimage import uniform_filter, generic_filter

# Function to add Gaussian noise with higher variance for better visibility
def add_gaussian_noise(image, mean=0, var=10.0):  # Higher variance for more noticeable noise  
    gaussian_noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_image = image + gaussian_noise
    return np.clip(noisy_image, 0, 255)

# Arithmetic mean filter using skimage's uniform filter
def arithmetic_mean_filter(image, kernel_size=3):
    return uniform_filter(image, size=kernel_size)

# Geometric mean filter using scipy's generic_filter
def geometric_mean_filter(image, kernel_size=3):
    def geo_mean(pixels):
        return np.exp(np.mean(np.log(pixels + 1e-7)))  # Avoiding log(0) with a small constant
    return generic_filter(image, geo_mean, size=kernel_size)

# Main function to load the image, add noise, and apply filters
def main():
    # Load the original image using PIL
    image_path = 'purple.jpg'  # Replace with the actual image path
    original_image = Image.open(image_path)
    
    if original_image is None:
        print("Error: Image not found.")
        return
    
    # Convert to grayscale using PIL
    grayscale_image = original_image.convert('L')  # 'L' mode is grayscale
    
    # Convert image to NumPy array for easier manipulation
    grayscale_image_np = np.array(grayscale_image)
    
    # Add Gaussian noise to the grayscale image with higher variance for better visibility
    noisy_image = add_gaussian_noise(grayscale_image_np, var=10.0)  # Increased variance for more visible noise
    
    # Apply 3x3 arithmetic mean filter (using skimage)
    arithmetic_filtered_image = arithmetic_mean_filter(noisy_image, kernel_size=3)
    
    # Apply 3x3 geometric mean filter (using scipy's generic_filter)
    geometric_filtered_image = geometric_mean_filter(noisy_image, kernel_size=3)
    
    # Display the images using matplotlib (with slightly larger output images)
    plt.figure(figsize=(16, 8))  # Increased size for better visibility
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(grayscale_image_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Gaussian Noisy Image (High Noise)')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Arithmetic Mean Filter (3x3)')
    plt.imshow(arithmetic_filtered_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Geometric Mean Filter (3x3)')
    plt.imshow(geometric_filtered_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()  # Adjust the spacing between the images
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
