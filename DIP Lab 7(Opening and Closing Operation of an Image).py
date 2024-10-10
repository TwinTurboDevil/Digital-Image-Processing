import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.morphology import opening, closing, square

# Read the RGB image
img_color = io.imread('purple.jpg')

# Define the structuring element (e.g., a 5x5 square)
se = square(5)

# Function to apply opening and closing on each color channel independently
def apply_morphological_operations(image, se):
    # Create empty arrays to hold the processed color channels
    opened_img = np.zeros_like(image)
    closed_img = np.zeros_like(image)

    # Apply the operations on each channel (R, G, B)
    for i in range(3):  # Loop over the R, G, B channels
        opened_img[:, :, i] = opening(image[:, :, i], se)
        closed_img[:, :, i] = closing(image[:, :, i], se)
    
    return opened_img, closed_img

# Apply opening and closing on the color image
opened_img_color, closed_img_color = apply_morphological_operations(img_color, se)

# Display the original image, opened image, and closed image
plt.figure(figsize=(15, 5))

# Plot original color image
plt.subplot(1, 3, 1)
plt.imshow(img_color)
plt.title('Original RGB Image')
plt.axis('off')

# Plot opened RGB image
plt.subplot(1, 3, 2)
plt.imshow(opened_img_color)
plt.title('Opened RGB Image')
plt.axis('off')

# Plot closed RGB image
plt.subplot(1, 3, 3)
plt.imshow(closed_img_color)
plt.title('Closed RGB Image')
plt.axis('off')

plt.tight_layout()
plt.show()
