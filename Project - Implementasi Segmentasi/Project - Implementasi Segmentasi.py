import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy.ndimage import sobel

def sobel_edge_detection(image):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges = (edges / edges.max()) * 255  # Normalize edges to 0-255
    return edges.astype(np.uint8)

def thresholding(image, threshold):
    # Apply thresholding using np.where for cleaner code
    return np.where(image > threshold, 255, 0).astype(np.uint8)

def main():
    input_image_path = 'C:/Users/ASUS/Documents/KULIAH/Smester 5/Pengolahan citra digital/Project - Implementasi Segmentasi/Everest mountain.jpg'
    
    # Read the image in grayscale mode
    image = imread(input_image_path, mode='L')  # Use mode='L' for grayscale
    
    # Check if the image is loaded correctly
    if image is None:
        print("Error loading image. Please check the file path.")
        return
    
    # Step 1: Perform Sobel edge detection
    edges = sobel_edge_detection(image)

    # Step 2: Perform basic thresholding on the edge-detected image
    threshold_value = 128  # Example threshold value, can be tuned
    thresholded_image = thresholding(edges, threshold_value)

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Sobel Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Thresholded Image')
    plt.imshow(thresholded_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Analysis
    print("Analysis:")
    print(f"Threshold value used: {threshold_value}")
    print(f"Number of edge pixels (post-thresholding): {np.sum(thresholded_image == 255)}")
    print(f"Number of non-edge pixels (post-thresholding): {np.sum(thresholded_image == 0)}")

if __name__ == "__main__":
    main()
