import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    default_file = 'night.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Perform Hough Circle Transform
    rows = gray.shape[0]
    # Adjust these parameters to detect more circles
    circles = cv.HoughCircles(
        gray, 
        cv.HOUGH_GRADIENT, 
        dp=1, 
        minDist=rows / 8, 
        param1=100, 
        param2=25,  # Lower this to detect more circles
        minRadius=5,  # Adjusted to allow smaller circles
        maxRadius=50  # Adjusted to allow larger circles
    )

    # Create a copy of the original image to draw circles
    output_image = src.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # Circle center in red
            cv.circle(output_image, center, 1, (0, 0, 255), 3)  # Red color for center
            # Circle outline in red
            radius = i[2]
            cv.circle(output_image, center, radius, (0, 0, 255), 3)  # Red color for outline

    # Convert images from BGR to RGB for display
    src_rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    output_rgb = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

    # Display the images using Matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(src_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Detected Circles')
    plt.imshow(output_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
