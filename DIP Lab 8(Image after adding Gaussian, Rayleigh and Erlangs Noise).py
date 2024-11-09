import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def add_rayleigh_noise(image, scale=0.1):
    rayleigh = np.random.rayleigh(scale, image.shape)
    noisy_image = image + rayleigh
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def add_erlang_noise(image, shape=2, scale=0.1):
    erlang = np.random.gamma(shape, scale, image.shape)
    noisy_image = image + erlang
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def main():
    image_path = 'purple.jpg'
    image = Image.open(image_path).convert('RGB')
    image = np.array(image) / 255.0  # Normalize to [0, 1]

    # Add noise multiple times to increase the effect
    noisy_image_gaussian = image
    noisy_image_rayleigh = image
    noisy_image_erlang = image

    for _ in range(3):  # Add noise 3 times
        noisy_image_gaussian = add_gaussian_noise(noisy_image_gaussian)
        noisy_image_rayleigh = add_rayleigh_noise(noisy_image_rayleigh)
        noisy_image_erlang = add_erlang_noise(noisy_image_erlang)

    # Convert back to [0, 255] and uint8
    noisy_image_gaussian = (noisy_image_gaussian * 255).astype(np.uint8)
    noisy_image_rayleigh = (noisy_image_rayleigh * 255).astype(np.uint8)
    noisy_image_erlang = (noisy_image_erlang * 255).astype(np.uint8)
    original_image = (image * 255).astype(np.uint8)

    # Display the images
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('Gaussian Noise')
    plt.imshow(noisy_image_gaussian)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Rayleigh Noise')
    plt.imshow(noisy_image_rayleigh)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('Erlang Noise')
    plt.imshow(noisy_image_erlang)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()