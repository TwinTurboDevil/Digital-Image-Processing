# Image enhancement using contrast stretching and histograms of both original and stretched image 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the color image
image = Image.open('purple.jpg')
image_array = np.array(image)

# Step 2: Split the image into R, G, B channels
r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]

# Step 3: Apply contrast stretching to each channel
def contrast_stretch(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)
    stretched_channel = (channel - min_val) / (max_val - min_val) * 255
    return stretched_channel.astype(np.uint8)

r_stretched = contrast_stretch(r)
g_stretched = contrast_stretch(g)
b_stretched = contrast_stretch(b)

# Print min and max values for debugging
print("Original R,G,B Channel - Min R: ", np.min(r), "Max R: ", np.max(r), "Min G: ", np.min(g), "Max G: ", np.max(g), "Min B: ", np.min(b), "Max B: ", np.max(b))
print("Stretched R Channel - Min: ", np.min(r_stretched), "Max: ", np.max(r_stretched))
print("Stretched G Channel - Min: ", np.min(g_stretched), "Max: ", np.max(g_stretched))
print("Stretched B Channel - Min: ", np.min(b_stretched), "Max: ", np.max(b_stretched))

# Step 4: Merge the stretched channels back together
stretched_image_array = np.stack((r_stretched, g_stretched, b_stretched), axis=2)
stretched_image = Image.fromarray(stretched_image_array)

# Step 5: Display the original and contrast-stretched images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Contrast-Stretched Image')
plt.imshow(stretched_image_array)
plt.axis('off')

plt.show()

# Step 6: Save the contrast-stretched image
stretched_image.save('contrast_stretched_color_purple.jpeg')

# Plot image and histogram
def plot_image_and_histogram(image_array, title):
    # Flatten the entire image array to get all pixel values
    all_pixels = image_array.ravel()

    plt.figure(figsize=(15, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')
    
    # Plot the histogram of all pixel values
    plt.subplot(1, 2, 2)
    plt.hist(all_pixels, bins=256, color='gray', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Plot the original image and their histograms
plot_image_and_histogram(image_array, 'Original Image')

# Plot the contrast-stretched image and its histogram
plot_image_and_histogram(stretched_image_array, 'Contrast-Stretched Image')
