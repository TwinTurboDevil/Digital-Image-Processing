import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def split_rgb_channels(image_path):
    # Open the RGB image
    rgb_image = Image.open(image_path).convert("RGB")

    # Convert the image to a NumPy array
    rgb_array = np.array(rgb_image)

    # Split the RGB channels 
    red_channel = rgb_array[..., 0]
    green_channel = rgb_array[..., 1]
    blue_channel = rgb_array[..., 2]

    return red_channel, green_channel, blue_channel

# Example usage
input_image_path = 'purple.jpg'  # Path to your RGB image
red_channel, green_channel, blue_channel = split_rgb_channels(input_image_path)

# Display the images using Matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(Image.open(input_image_path))
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(red_channel, cmap='Reds')
plt.title("Red Channel")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(green_channel, cmap='Greens')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(blue_channel, cmap='Blues')
plt.title("Blue Channel")
plt.axis('off')

plt.show()
