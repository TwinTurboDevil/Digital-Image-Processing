import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.morphology import erosion, square

# Load the image 'tree.jpeg'
image = io.imread('purple.jpg')

# Define a structuring element (e.g., a 3x3 square)
structuring_element = square(3)

# Function to apply erosion on an RGB image
def erosion_rgb(image, se):
    # Create an empty array to hold the eroded image
    eroded_image = np.zeros_like(image)
    
    # Apply erosion to each color channel independently (R, G, B)
    for i in range(3):  # Loop over the R, G, B channels
        eroded_image[:, :, i] = erosion(image[:, :, i], se)
    
    return eroded_image

# Apply erosion to the RGB image
eroded_rgb_image = erosion_rgb(image, structuring_element)

# Display the original and eroded RGB images
plt.figure(figsize=(12, 6))

# Plot original RGB image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original RGB Image')
plt.axis('off')

# Plot eroded RGB image
plt.subplot(1, 2, 2)
plt.imshow(eroded_rgb_image)
plt.title('Eroded RGB Image')
plt.axis('off')

plt.tight_layout()
plt.show()
