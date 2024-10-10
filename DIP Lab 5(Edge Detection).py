import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_laplace
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image and convert it to RGB."""
    return Image.open(image_path).convert('RGB')

def apply_roberts(image):
    """Apply Roberts operator."""
    roberts_operator = np.array([[1, 0], [0, -1]])
    edges = np.zeros(image.shape)

    for channel in range(3):
        edges_channel = np.abs(convolve(image[:, :, channel].astype(float), roberts_operator))
        edges[:, :, channel] = edges_channel

    return np.clip(np.sqrt(np.sum(edges ** 2, axis=2)), 0, 255)

def apply_sobel(image):
    """Apply Sobel operator."""
    sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_operator_y = sobel_operator_x.T
    edges = np.zeros(image.shape)

    for channel in range(3):
        edges_x = convolve(image[:, :, channel].astype(float), sobel_operator_x)
        edges_y = convolve(image[:, :, channel].astype(float), sobel_operator_y)
        edges_channel = np.sqrt(edges_x ** 2 + edges_y ** 2)
        edges[:, :, channel] = edges_channel

    return np.clip(np.sqrt(np.sum(edges ** 2, axis=2)), 0, 255)

def apply_prewitt(image):
    """Apply Prewitt operator."""
    prewitt_operator_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_operator_y = prewitt_operator_x.T
    edges = np.zeros(image.shape)

    for channel in range(3):
        edges_x = convolve(image[:, :, channel].astype(float), prewitt_operator_x)
        edges_y = convolve(image[:, :, channel].astype(float), prewitt_operator_y)
        edges_channel = np.sqrt(edges_x ** 2 + edges_y ** 2)
        edges[:, :, channel] = edges_channel

    return np.clip(np.sqrt(np.sum(edges ** 2, axis=2)), 0, 255)

def apply_isotropic(image):
    """Apply Isotropic operator (Laplacian of Gaussian)."""
    # Adjusting the kernel size and sigma for better edge detection
    h = gaussian_laplace(np.zeros((5, 5)), sigma=1)  # Experiment with different sigma
    edges = np.zeros(image.shape)

    for channel in range(3):
        edges_channel = np.abs(convolve(image[:, :, channel].astype(float), h))
        edges[:, :, channel] = edges_channel

    return np.clip(np.sqrt(np.sum(edges ** 2, axis=2)), 0, 255)

# Main code execution
image_path = 'night.jpg'  # Change to your image path
input_image = load_image(image_path)
input_image_np = np.array(input_image)

# Apply edge detection operators
edges_roberts = apply_roberts(input_image_np)
edges_sobel = apply_sobel(input_image_np)
edges_prewitt = apply_prewitt(input_image_np)
edges_isotropic = apply_isotropic(input_image_np)

# Threshold value for clarity (adjust as needed)
threshold = 0.05 * 255

# Apply thresholding
edges_roberts[edges_roberts < threshold] = 0
edges_sobel[edges_sobel < threshold] = 0
edges_prewitt[edges_prewitt < threshold] = 0
edges_isotropic[edges_isotropic < threshold] = 0

# Display results
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(input_image_np)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(edges_roberts, cmap='gray')
plt.title('Roberts Operator')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edges_sobel, cmap='gray')
plt.title('Sobel Operator')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(edges_prewitt, cmap='gray')
plt.title('Prewitt Operator')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edges_isotropic, cmap='gray')
plt.title('Isotropic Operator')
plt.axis('off')

# Adjust the layout
plt.suptitle('Edge Detection Results with Thresholding')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
