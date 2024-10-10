import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def isotropic_operator(image, sigma=1.0):
    # Create a Gaussian kernel
    size = int(2 * np.ceil(2 * sigma) + 1)
    x = np.linspace(-size//2, size//2, size)
    gauss_kernel = np.exp(-0.5 * (x**2 / sigma**2))
    gauss_kernel /= gauss_kernel.sum()  # Normalize

    # Create Laplacian of Gaussian (LoG) kernel
    x, y = np.meshgrid(x, x)
    log_kernel = (x**2 + y**2 - 2 * sigma**2) * np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    log_kernel /= log_kernel.sum()  # Normalize

    # Apply convolution using the LoG kernel
    edges = convolve(image, log_kernel)
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    return edges

# Load an image
image_path = 'purple.jpg'  # Change this to your image path
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_np = np.array(image)

# Apply Isotropic operator (Laplacian of Gaussian)
edges_isotropic = isotropic_operator(image_np)

# Display results using Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edges using Isotropic Operator (LoG)')
plt.imshow(edges_isotropic, cmap='gray')
plt.axis('off')

plt.show()
