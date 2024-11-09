import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt

def edge_detection(image):
    # Convert the image to grayscale
    gray_image = image.convert("L")
    # Apply a edge detection filter
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return edges

def hough_transform(edges):
    width, height = edges.size
    # Convert edges to binary array
    edge_array = np.array(edges) > 128  # Thresholding
    # Parameters for Hough Transform
    diag_len = int(np.sqrt(width ** 2 + height ** 2))
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    rhos = np.arange(-diag_len, diag_len, 1)
    
    # Hough accumulator
    accumulator = np.zeros((len(rhos), len(theta)), dtype=int)  # Use int instead of np.int

    # Perform Hough Transform
    y_indices, x_indices = np.nonzero(edge_array)
    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for t in range(len(theta)):
            r = int(x * np.cos(theta[t]) + y * np.sin(theta[t])) + diag_len
            accumulator[r, t] += 1

    return accumulator, theta, rhos

def draw_lines(image, accumulator, theta, rhos):
    draw = ImageDraw.Draw(image)
    threshold = np.max(accumulator) * 0.5  # Threshold for line detection
    for r in range(accumulator.shape[0]):
        for t in range(accumulator.shape[1]):
            if accumulator[r, t] > threshold:
                rho = rhos[r]
                theta_val = theta[t]
                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                draw.line((x1, y1, x2, y2), fill=255, width=2)

# Load the image
image_path = 'night.jpg'  # Replace with your image file path
image = Image.open(image_path)

# Display the original image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

# Perform edge detection
edges = edge_detection(image)

# Display the edges
plt.subplot(1, 3, 2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Perform Hough Transform
accumulator, theta, rhos = hough_transform(edges)

# Draw detected lines on the original image
drawn_image = image.copy()
draw_lines(drawn_image, accumulator, theta, rhos)

# Display the detected lines
plt.subplot(1, 3, 3)
plt.title('Detected Lines')
plt.imshow(drawn_image)
plt.axis('off')

plt.tight_layout()
plt.show()
