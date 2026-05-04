import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

if not os.path.exists('output_images'):
    os.makedirs('output_images')

cnt = 0

def load(urls = [
    'Week1/images/image1.jpg', 
    'Week1/images/landscape.jpg',
    'Week1/images/portrait.jpg',
]):
    images = []
    for url in urls:
        images.append(cv2.imread(url))

    return images


def box_blur(img):

    blur_kernel = np.ones((3, 3), dtype=float)

    blur_kernel /= 9

    blurred_img = cv2.filter2D(img, -1, blur_kernel)

    return 'Blur', blurred_img


def edge_detection(img):
        
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0 ,1]
        ], dtype=np.float32)

    sobel_y = sobel_x.T

    edge_x = cv2.filter2D(img, -1, sobel_x)
    edge_y = cv2.filter2D(img, -1, sobel_y)
    
    edges = edge_x.copy()
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            for c in range(3):
                edges[x, y, c] = max(edges[x, y, c], edge_y[x, y, c]) 

    return 'Edge Detection', edges


def sharpen(img):
    sharpen_kernel = np.zeros((3, 3), dtype=float)

    sharpen_kernel[0, 1] = -1
    sharpen_kernel[1, 0] = -1
    sharpen_kernel[1, 2] = -1
    sharpen_kernel[2, 1] = -1

    sharpen_kernel[1, 1] = 5

    sharpen_img = cv2.filter2D(img, -1, sharpen_kernel)

    return 'Sharpness', sharpen_img


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian(img):
    kernel_gaussian = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], np.float32) / 16
    gaussian_img = cv2.filter2D(img, -1, kernel_gaussian)

    return 'Gaussian Blur', gaussian_img


def laplacian(img):
    kernel_laplacian = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], np.float32)
    
    return 'Laplacian Edge', cv2.filter2D(img, -1 , kernel_laplacian)


def display(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def printPixelStatistics(img):
    # Print statistics
    print(f"Statistics for the image:")
    print(f"- Minimum Intensity: {np.min(img)}")
    print(f"- Maximum Intensity: {np.max(img)}")
    print(f"- Mean Intensity:    {np.mean(img):.2f}")
    print(f"- Std Deviation:     {np.std(img):.2f}")

def task(img):
    global cnt

    # 2. display image
    display(img)

    # 3. Print data
    print("Shape: ", img.shape)
    print("dtype: ", img.dtype)
    printPixelStatistics(img)

    # 3. Apply filters

    for i, apply_filter in enumerate([box_blur, gaussian, edge_detection, laplacian]):
        img_cp = img.copy()
        name, filtered_img = apply_filter(img_cp)

        cv2.imwrite(f'output_images/{name}{cnt}.jpg', filtered_img)

        cnt += 1

        plt.subplot(2, 3, i + 2)
        display_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB) if len(filtered_img.shape) == 3 else filtered_img
        plt.imshow(display_img, cmap='gray')
        plt.title(name)
        plt.axis('off')

    plt.show()

'''
Note: Change load image URL to work.
'''
if __name__ == '__main__':
    # 1. Load
    images = load()
    for image in images:
        task(image)

'''
Written Analysis:
Choosing the most effective filter depends on the image's content. 
For landscape photos the gaussian blur is best because it subtly reduces high-frequency noise and sensor grain while maintaining a natural, soft transition between colors, unlike the harsher box blur. 
For portraits, the sharpness kernel stands out by emphasizing facial features and eyes. 
In contrast, for technical or structural images, the Laplacian filter is most effective as it creates a high-contrast map of all edges, making it easier to analyze silhouettes or detect mechanical boundaries that are invisible in standard color mode.
'''