import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pixel_skipping_resize(image_array, new_width, new_height):
    """
    Resize an image using the pixel skipping method.

    Parameters:
    image_array (numpy.ndarray): Input image array.
    new_width (int): Desired width of the resized image.
    new_height (int): Desired height of the resized image.

    Returns:
    numpy.ndarray: Resized image array using pixel skipping.
    """
    old_height, old_width, channels = image_array.shape

    # Create an empty array for the resized image
    resized_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)

    # Calculate the scaling factors
    row_scale = old_height / new_height
    col_scale = old_width / new_width

    # Fill in the resized image array
    for y in range(new_height):
        for x in range(new_width):
            # Find the corresponding pixel in the original image
            src_y = int(y * row_scale)
            src_x = int(x * col_scale)

            # Ensure indices are within bounds
            src_y = min(src_y, old_height - 1)
            src_x = min(src_x, old_width - 1)

            # Copy the pixel value from the original image
            resized_image_array[y, x] = image_array[src_y, src_x]

    return resized_image_array

# Load an image using PIL
image_path = 'purple.jpg'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Get and print the original image dimensions
original_height, original_width, _ = image_array.shape
print(f"Original image dimensions: {original_width}x{original_height}")

# Define new dimensions for resizing (for demonstration purposes, these can be changed)
new_width = int(input("Enter the new width: "))  # Input new width
new_height = int(input("Enter the new height: "))  # Input new height

# Resize the image using the pixel skipping method
resized_image_array = pixel_skipping_resize(image_array, new_width, new_height)

# Convert the result back to an image
resized_image = Image.fromarray(resized_image_array)

# Save the result
output_path = 'pixel_skipping_resized_purple.jpg'
resized_image.save(output_path)

# Display the images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')
plt.text(0.5, -0.1, f"Dimensions: {original_width}x{original_height}", ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

# Resized Image
plt.subplot(1, 2, 2)
plt.title('Resized Image using Pixel Skipping')
plt.imshow(resized_image_array)
plt.axis('off')
plt.text(0.5, -0.1, f"Dimensions: {new_width}x{new_height}", ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
