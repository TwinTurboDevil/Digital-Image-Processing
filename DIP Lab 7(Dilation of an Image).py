import numpy as np
import cv2
import matplotlib.pyplot as plt

def dilate_image(image_path, kernel_size=(5,5), iterations=1):
    # Read the image in RGB
    image = cv2.imread(image_path)
    # Convert from BGR to RGB (OpenCV reads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a kernel for dilation
    kernel = np.ones(kernel_size, np.uint8)
    
    # Perform dilation on each channel
    dilated = np.zeros_like(image)
    for i in range(3):  # For each color channel
        dilated[:,:,i] = cv2.dilate(image[:,:,i], kernel, iterations=iterations)
    
    return image, dilated

def plot_results(original, dilated):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(dilated)
    ax2.set_title('Dilated Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Path to the image file
image_path = 'purple.jpg'  # Replace with the path to your image

# Perform dilation
original, dilated = dilate_image(image_path, kernel_size=(5,5), iterations=1)

# Plot results
plot_results(original, dilated)