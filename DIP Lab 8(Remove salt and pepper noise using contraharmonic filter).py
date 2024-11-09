import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image, amount=0.02):
    noisy_image = random_noise(image, mode='s&p', amount=amount)
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    return noisy_image

# Contraharmonic filter function
def contraharmonic_filter(image, kernel_size=3, Q=1.5):
    # Get the dimensions of the image
    height, width = image.shape
    
    # Create an empty image to store the output
    filtered_image = np.zeros_like(image, dtype=np.float64)
    
    # Define the kernel radius
    offset = kernel_size // 2
    
    # Apply the contraharmonic filter to each pixel
    for i in range(height):
        for j in range(width):
            # Define the local neighborhood
            min_i, max_i = max(0, i - offset), min(height, i + offset + 1)
            min_j, max_j = max(0, j - offset), min(width, j + offset + 1)
            region = image[min_i:max_i, min_j:max_j]
            
            # Calculate the numerator and denominator for the contraharmonic filter
            try:
                numerator = np.sum(region**(Q + 1))
                denominator = np.sum(region**Q)
                
                # Handle case where denominator is zero (avoid division by zero)
                if denominator == 0:
                    # If denominator is zero, handle gracefully by assigning median value of the region
                    filtered_image[i, j] = np.median(region)
                else:
                    # Otherwise, apply the filter formula
                    filtered_image[i, j] = numerator / denominator
            except ZeroDivisionError:
                # In case of division by zero error, assign median value
                filtered_image[i, j] = np.median(region)
            except ValueError:
                # In case of NaN or inf values in calculations, assign median value
                filtered_image[i, j] = np.median(region)
    
    return np.uint8(np.clip(filtered_image, 0, 255))

# Main function to load the image, add noise, and apply filters
def main():
    # Load the original image
    image_path = 'night.jpg'  # Update this path if necessary
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Image not found.")
        return
    
    # Add salt and pepper noise to the image
    noisy_image = add_salt_and_pepper_noise(original_image, amount=0.05)  # 5% noise
    
    # Apply Contraharmonic filter with Q = 1.5
    contraharmonic_filtered_image_Q1_5 = contraharmonic_filter(noisy_image, kernel_size=3, Q=1.5)
    
    # Apply Contraharmonic filter with Q = -1.5
    contraharmonic_filtered_image_Qm1_5 = contraharmonic_filter(noisy_image, kernel_size=3, Q=-1.5)
    
    # Display the images using matplotlib
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Salt and Pepper Noisy Image')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Contraharmonic Filter Q=1.5')
    plt.imshow(contraharmonic_filtered_image_Q1_5, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Contraharmonic Filter Q=-1.5')
    plt.imshow(contraharmonic_filtered_image_Qm1_5, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
