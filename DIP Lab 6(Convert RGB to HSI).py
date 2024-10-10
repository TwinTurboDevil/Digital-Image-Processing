import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_hsi(rgb_image):
    # Normalize the RGB values to the range [0, 1]
    rgb_image = rgb_image / 255.0
    
    # Extract R, G, B channels
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    
    # Calculate Intensity
    I = (R + G + B) / 3.0
    
    # Calculate Saturation
    num = 1 - (np.minimum(R, np.minimum(G, B)) / (R + G + B + 1e-6))  # Adding a small epsilon to avoid division by zero
    S = 1 - num
    
    # Calculate Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B) + 1e-6)  # Adding epsilon to avoid division by zero
    theta = np.arccos(num / den)
    
    H = np.zeros_like(I)
    
    # Hue calculation
    H[B <= G] = theta[B <= G]
    H[B > G] = 2 * np.pi - theta[B > G]
    
    # Normalize Hue to [0, 1]
    H = H / (2 * np.pi)
    
    # Stack H, S, I into a single HSI image
    hsi_image = np.stack((H, S, I), axis=-1)
    
    return hsi_image

# Load an RGB image using PIL
image_path = 'purple.jpg'  # Change this to your image path
rgb_image = Image.open(image_path)
rgb_image = np.array(rgb_image)

# Convert the RGB image to HSI
hsi_image = rgb_to_hsi(rgb_image)

# Extract H, S, I components
H = hsi_image[:, :, 0]
S = hsi_image[:, :, 1]
I = hsi_image[:, :, 2]

# Plotting the results
plt.figure(figsize=(16, 8))

# Original RGB Image
plt.subplot(2, 3, 1)
plt.title('RGB Image')
plt.imshow(rgb_image)
plt.axis('off')

# H Channel (Hue)
plt.subplot(2, 3, 2)
plt.title('Hue Channel')
plt.imshow(H, cmap='hsv')
plt.axis('off')

# S Channel (Saturation)
plt.subplot(2, 3, 3)
plt.title('Saturation Channel')
plt.imshow(S, cmap='gray')
plt.axis('off')

# I Channel (Intensity)
plt.subplot(2, 3, 4)
plt.title('Intensity Channel')
plt.imshow(I, cmap='gray')
plt.axis('off')

# HSI Image (composite)
plt.subplot(2, 3, 5)
plt.title('HSI Image')
plt.imshow(hsi_image)
plt.axis('off')

plt.tight_layout()
plt.show()
