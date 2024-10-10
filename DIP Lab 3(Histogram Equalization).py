from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to plot the histogram in bar chart format
def plot_histogram(image, title):
    # Convert the image to a numpy array and flatten all pixel intensities across the RGB channels
    img_flatten = image.flatten()
    
    # Plot histogram with 256 bins (for pixel values 0-255)
    plt.hist(img_flatten, bins=256, range=[0, 256], color='gray', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Function to apply histogram equalization to each channel
def equalize_histogram_color(image):
    img_array = np.array(image)
    equalized_img_array = np.zeros_like(img_array)
    
    for i in range(3):  # Loop over each channel (R, G, B)
        # Flatten the channel, compute histogram and cumulative distribution function (CDF)
        channel = img_array[:, :, i]
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]  # Normalize the CDF
        
        # Use linear interpolation to find new pixel values
        equalized_channel = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)
        equalized_img_array[:, :, i] = equalized_channel
    
    return Image.fromarray(equalized_img_array.astype('uint8'))

# Load the image using PIL
image = Image.open('purple.jpg')

# Apply histogram equalization
equalized_image = equalize_histogram_color(image)

# Convert images to numpy arrays for histogram plotting
original_image_np = np.array(image)
equalized_image_np = np.array(equalized_image)

# Plot original image and its histogram
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plot_histogram(original_image_np, 'Original Histogram')

# Plot equalized image and its histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image)
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plot_histogram(equalized_image_np, 'Equalized Histogram')

plt.tight_layout()
plt.show()
