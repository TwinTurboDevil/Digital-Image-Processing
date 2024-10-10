import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_grayscale(image_path):
    # Open the RGB image
    rgb_image = Image.open(image_path).convert("RGB")

    # Convert the image to a NumPy array
    rgb_array = np.array(rgb_image)

    # Calculate the grayscale values using the luminosity method
    # Grayscale = 0.2989*R + 0.5870*G + 0.1140*B
    grayscale_array = np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Convert the grayscale array to uint8 type
    grayscale_image = Image.fromarray(grayscale_array.astype(np.uint8), 'L')

    return grayscale_image

# Example usage
input_image_path = 'purple.jpg'  # Path to your RGB image
grayscale_image = rgb_to_grayscale(input_image_path)

# Display the images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Image.open(input_image_path))
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(grayscale_image, cmap='gray')
plt.title("Converted Grayscale Image")
plt.axis('off')

plt.show()
