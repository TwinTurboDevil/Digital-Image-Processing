from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to add periodic noise (similar to the second code)
def add_periodic_noise(image, frequency, amplitude):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Generate sine wave for periodic noise
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
    X, Y = np.meshgrid(x, y)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * X / cols)

    # Add the sine wave to the image
    noisy_image = image + sine_wave  # Single-channel image
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to apply band reject filter (to remove periodic noise)
def band_reject_filter(image, low_radius, high_radius):
    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Perform Fourier Transform
    F = np.fft.fftshift(np.fft.fft2(image))  # Shift zero frequency to center

    # Create a band reject filter
    filter_mask = np.ones((rows, cols), np.float32)  # Initialize filter with ones

    # Set the band reject filter to 0 within the specified radius
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if low_radius < distance < high_radius:
                filter_mask[i, j] = 0  # Reject frequencies in the specified band

    # Apply the filter to the Fourier-transformed image
    F_filtered = F * filter_mask

    # Inverse Fourier Transform to get back to the spatial domain
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    return np.uint8(np.clip(filtered_image, 0, 255))

# Main function to load the image, add noise, apply the band reject filter, and display results
def main():
    # Load the original image using PIL
    image_path = 'flower.jpg'  # Replace with the actual image path
    original_image = Image.open(image_path)

    # Convert to grayscale
    original_image = original_image.convert('L')

    # Convert image to NumPy array
    original_image_np = np.array(original_image)

    # Add periodic noise to the image using sine wave
    frequency = 50  # Frequency of the noise (in the frequency domain)
    amplitude = 100  # Amplitude of the noise (increase amplitude to make it visible)
    noisy_image = add_periodic_noise(original_image_np, frequency, amplitude)

    # Define the band reject filter parameters (you can adjust the radius values)
    low_radius = 30  # Minimum radius of the band to reject
    high_radius = 60  # Maximum radius of the band to reject

    # Apply the band reject filter to remove periodic noise
    filtered_image = band_reject_filter(noisy_image, low_radius, high_radius)

    # Convert the filtered image back to PIL image
    filtered_image_pil = Image.fromarray(filtered_image)

    # Display the images using matplotlib
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Image with Periodic Noise')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Filtered Image (Band Reject)')
    plt.imshow(filtered_image_pil, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
