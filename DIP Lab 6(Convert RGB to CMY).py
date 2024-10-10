import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_cmy(image_path):
    # Open the RGB image
    rgb_image = Image.open(image_path).convert("RGB")

    # Convert the image to a NumPy array
    rgb_array = np.array(rgb_image)

    # Convert RGB to CMY
    cmy_array = 255 - rgb_array

    # Create a new CMY image from the CMY array
    cmy_image = Image.fromarray(cmy_array.astype(np.uint8), 'RGB')

    return cmy_image

# Example usage
input_image_path = 'purple.jpg'  # Path to your RGB image
cmy_image = rgb_to_cmy(input_image_path)

# Save the CMY image
output_image_path = 'purple_cmy.jpg'  # Path to save the CMY image
cmy_image.save(output_image_path)

# Display the images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Image.open(input_image_path))
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cmy_image)
plt.title("Converted CMY Image")
plt.axis('off')

plt.show()
