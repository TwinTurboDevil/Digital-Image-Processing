import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def cepstrum_homomorphic_transform(image_path):
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    u = np.array(img, dtype=np.float32)

    # Step 1: Apply DFT
    v = fft2(u)
    v = fftshift(v)  # Shift zero frequency to center

    # Step 2: Take log of magnitude and add phase
    v_log_magnitude = np.log(np.abs(v) + 1)  # Add 1 to avoid log(0)
    v_phase = np.angle(v)
    s = v_log_magnitude * np.exp(1j * v_phase)

    # Step 3: Apply inverse DFT
    c = np.real(ifft2(ifftshift(s)))

    return u, np.abs(v), np.abs(s), c

def plot_results(u, v, s, c):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(u, cmap='gray')
    axs[0].set_title('u(i,j)-Input image')
    axs[0].axis('off')
    
    axs[1].imshow(np.log(v + 1), cmap='gray')  # Log for better visualization
    axs[1].set_title('v(k,l)-After applying DFT')
    axs[1].axis('off')
    
    axs[2].imshow(s, cmap='gray')
    axs[2].set_title('s(k,l)-After log and phase transform')
    axs[2].axis('off')
    
    axs[3].imshow(c, cmap='gray')
    axs[3].set_title('c(i,j)-Final output after applying Inverse DFT')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Path to the image file
image_path = 'purple.jpg'  # Replace with the path to your image

# Apply Cepstrum (Homomorphic Transform)
u, v, s, c = cepstrum_homomorphic_transform(image_path)

# Plot results
plot_results(u, v, s, c)