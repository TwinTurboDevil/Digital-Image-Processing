# Convert a digital image into a negative image

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image = Image.open('purple.jpg')

# Step 2: Convert image to a NumPy array
image_array = np.array(image)

# Step 3: Invert image values (convert to negative)
negative_array = 255 - image_array

# Step 4: Convert the negative array back to a PIL Image object (optional)
negative_image = Image.fromarray(negative_array.astype(np.uint8))

# Step 5: Display the original and negative images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Negative Image')
plt.imshow(negative_array, cmap='gray')  # Use grayscale colormap
plt.axis('off')

plt.show()

# Step 6: (Optional) Save the negative image
negative_image.save('negative_purple.jpg')  # Use the PIL Image if needed