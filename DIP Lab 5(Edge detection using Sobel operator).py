import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def sobel_operator(image):
    # Define Sobel kernels
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply convolution using the defined kernels
    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)

    # Calculate the magnitude of gradients
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude

# Load an image
image_path = 'purple.jpg'  # Change this to your image path
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_np = np.array(image)

# Apply Sobel operator
edges = sobel_operator(image_np)

# Display results using Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edges using Sobel Operator')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
