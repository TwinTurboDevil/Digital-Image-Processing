import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rgb_to_yiq(rgb_image):
    rgb_image = rgb_image / 255.0
    Y = 0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
    I = 0.596 * rgb_image[:, :, 0] - 0.274 * rgb_image[:, :, 1] - 0.321 * rgb_image[:, :, 2]
    Q = 0.211 * rgb_image[:, :, 0] - 0.523 * rgb_image[:, :, 1] + 0.312 * rgb_image[:, :, 2]
    yiq_image = np.stack((Y, I, Q), axis=-1)
    return yiq_image

# Load an RGB image using PIL
image_path = 'purple.jpg'  # Change this to your image path
rgb_image = Image.open(image_path)
rgb_image = np.array(rgb_image)

# Convert the RGB image to YIQ
yiq_image = rgb_to_yiq(rgb_image)

# Plotting the original RGB image and the Y, I, and Q channels
plt.figure(figsize=(16, 4))

# Original RGB Image
plt.subplot(1, 4, 1)
plt.title('Original RGB Image')
plt.imshow(rgb_image)
plt.axis('off')

# Y Channel
plt.subplot(1, 4, 2)
plt.title('Y Channel (Luminance)')
plt.imshow(yiq_image[:, :, 0], cmap='gray')
plt.axis('off')

# I Channel
plt.subplot(1, 4, 3)
plt.title('I Channel (Chrominance I)')
plt.imshow(yiq_image[:, :, 1], cmap='gray')
plt.axis('off')

# Q Channel
plt.subplot(1, 4, 4)
plt.title('Q Channel (Chrominance Q)')
plt.imshow(yiq_image[:, :, 2], cmap='gray')
plt.axis('off')

plt.show()
