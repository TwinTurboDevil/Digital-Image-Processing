import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def linear_interpolation_resize(image_array, new_width, new_height):
    """
    Resize an image using linear interpolation method.

    Parameters:
    image_array (numpy.ndarray): Input image array.
    new_width (int): Desired width of the resized image.
    new_height (int): Desired height of the resized image.

    Returns:
    numpy.ndarray: Resized image array using linear interpolation.
    """
    old_height, old_width, channels = image_array.shape

    # Create an empty array for the resized image
    resized_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)

    # Calculate the scaling factors
    row_scale = old_height / new_height
    col_scale = old_width / new_width

    # Iterate over each pixel in the new image
    for y in range(new_height):
        for x in range(new_width):
            # Find the corresponding coordinates in the original image
            orig_y = y * row_scale
            orig_x = x * col_scale

            # Calculate the coordinates of the four surrounding pixels
            y0 = int(orig_y)
            x0 = int(orig_x)
            y1 = min(y0 + 1, old_height - 1)
            x1 = min(x0 + 1, old_width - 1)

            # Calculate the weights for interpolation
            dy = orig_y - y0
            dx = orig_x - x0
            w00 = (1 - dx) * (1 - dy)
            w01 = dx * (1 - dy)
            w10 = (1 - dx) * dy
            w11 = dx * dy

            # Interpolate the pixel values
            for c in range(channels):
                pixel_value = (w00 * image_array[y0, x0, c] +
                               w01 * image_array[y0, x1, c] +
                               w10 * image_array[y1, x0, c] +
                               w11 * image_array[y1, x1, c])
                resized_image_array[y, x, c] = pixel_value

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

# Resize the image using the linear interpolation method
resized_image_array = linear_interpolation_resize(image_array, new_width, new_height)

# Convert the result back to an image
resized_image = Image.fromarray(np.uint8(resized_image_array))

# Save the result
output_path = 'linear_interpolation_resized_purple.jpg'
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
plt.title('Resized Image using Linear Interpolation')
plt.imshow(resized_image_array.astype(np.uint8))
plt.axis('off')
plt.text(0.5, -0.1, f"Dimensions: {new_width}x{new_height}", ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
