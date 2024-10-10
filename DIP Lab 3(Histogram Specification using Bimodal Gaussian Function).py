from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to plot the histogram in bar chart format
def plot_histogram(image, title):
    # Flatten all pixel intensities across the RGB channels
    img_flatten = image.flatten()
    
    # Plot histogram with 256 bins (for pixel values 0-255)
    plt.hist(img_flatten, bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Function to convert the image to single intensity by averaging the RGB channels
def rgb_to_intensity(image):
    return np.mean(image, axis=2).astype(np.uint8)

# Function to apply the bimodal Gaussian distribution to generate a target histogram
def bimodal_gaussian_target():
    x = np.linspace(0, 255, 256)
    
    # Bimodal Gaussian: two normal distributions combined
    gaussian1 = norm.pdf(x, loc=70, scale=15)
    gaussian2 = norm.pdf(x, loc=180, scale=25)
    
    target_hist = gaussian1 + gaussian2
    target_hist = target_hist / target_hist.sum()  # Normalize
    
    return target_hist

# Function to modify each channel of the image based on the target bimodal Gaussian distribution
def modify_image_to_target(image, target_hist):
    img_array = np.array(image)
    modified_img_array = np.zeros_like(img_array)
    
    for i in range(3):  # Loop over each channel (R, G, B)
        # Calculate the cumulative distribution function (CDF) for each channel
        channel = img_array[:, :, i]
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf_image = hist.cumsum()
        cdf_image = cdf_image / cdf_image[-1]  # Normalize
        
        # CDF of target distribution
        cdf_target = np.cumsum(target_hist)
        
        # Mapping the original channel to match the target distribution
        channel_equalized = np.interp(channel.flatten(), bins[:-1], cdf_image)
        channel_mapped = np.interp(channel_equalized, cdf_image, cdf_target * 255)
        
        modified_img_array[:, :, i] = channel_mapped.reshape(channel.shape)
    
    return Image.fromarray(modified_img_array.astype('uint8'))

# Load the image using PIL
image = Image.open('purple.jpg')
image_np = np.array(image)

# Display original image and its histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plot_histogram(image_np, "Original Histogram")

# Generate bimodal Gaussian target histogram
target_hist = bimodal_gaussian_target()

# Plot the target bimodal Gaussian distribution (as a bar chart)
plt.figure(figsize=(6, 4))
plt.bar(np.linspace(0, 255, 256), target_hist, width=1, color='r', alpha=0.7)
plt.title("Target Bimodal Gaussian Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Modify the image based on the target distribution
modified_image = modify_image_to_target(image, target_hist)

# Display modified image and its histogram
plt.figure(figsize=(10, 5))

# Plot modified image
plt.subplot(1, 2, 1)
plt.imshow(modified_image)
plt.title("Modified Image")

# Plot histogram of modified image
plt.subplot(1, 2, 2)
plot_histogram(np.array(modified_image), "Modified Image Histogram")

plt.tight_layout()
plt.show()
