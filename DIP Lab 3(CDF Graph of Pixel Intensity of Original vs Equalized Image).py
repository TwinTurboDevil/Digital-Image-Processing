from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to convert the image to single intensity by averaging the RGB channels
def rgb_to_intensity(image):
    return np.mean(image, axis=2).astype(np.uint8)

# Function to calculate the cumulative distribution function (CDF)
def calculate_cdf(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to the range [0, 1]
    return cdf_normalized

# Function to apply histogram equalization to the intensity image
def equalize_histogram(image):
    img_array = np.array(image)
    
    # Compute histogram and cumulative distribution function (CDF)
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize the CDF
    
    # Use linear interpolation to find new pixel values
    equalized_img_array = np.interp(img_array.flatten(), bins[:-1], cdf_normalized).reshape(img_array.shape)
    
    return equalized_img_array

# Load the image using PIL
image = Image.open('purple.jpg')
original_image_np = np.array(image)

# Convert the RGB image to single intensity values
intensity_image = rgb_to_intensity(original_image_np)

# Apply histogram equalization
equalized_intensity_image = equalize_histogram(intensity_image)

# Calculate CDFs for both the original and equalized images
cdf_original = calculate_cdf(intensity_image)
cdf_equalized = calculate_cdf(equalized_intensity_image)

# Create a CDF graph
plt.figure(figsize=(6, 6))

# Plot the original CDF on the x-axis and the equalized CDF on the y-axis
plt.plot(cdf_original, cdf_equalized, 'b-', alpha=0.8)

plt.title('CDF: Original vs Equalized Pixel Intensity')
plt.xlabel('Original Image CDF')
plt.ylabel('Equalized Image CDF')
plt.grid(True)
plt.show()
