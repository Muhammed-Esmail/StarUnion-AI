'''
# Theory Section

- What is a pixel? Explain intensity values for 8-bit images.
    
    Pixel (picture element) is a single 2D unit of color in an image. It is the building blocks of images.
    8-bit images refer to how many bits are used to describe a pixel color.

    For Grayscale images:
    - 8-bit images use 8-bits per pixel (256 shades of gray)
    For Color images:
    - 8-bit images use 24-bits per pixel (8 for R, 8 for G and 8 for B)

- What is the shape of a color image with H=480, W=640? Explain each dimension.

    The image has 480 rows and 640 columns of pixels with 3 channels for colors per pixel.

- Explain convolution step by step. Include kernel, sliding window, and output.

    Starting from the top-left corner (convention), the kernel slides from left to right/up to down and extracts the output values from the weighted sum of the kernel into the output map.
    
- Compare Box Blur vs Gaussian Blur vs Sobel — purpose and kernel differences.

    Box blur: attempts to blur out the details of the image by taking the average of the colors in a specified radius. All weights are equal. This however results in a somewhat blocky image.
    Gaussian blur: attempts to blur out the details of the image by taking normally distributed values as weights. Giving higher weights to closer pixels. This results in a more smooth blur compared to box blur.
    Sobel: attempts to detect edges in the image by detecting a significant change in pixel color values. The kernel is a negative vs positive weights around the axis in question to take the difference. 
'''

import numpy as np
import cv2

def load(url = 'Week1\\images\\landscape.jpg'):
    img = cv2.imread(url)
    return img


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
            edges[x, y] = max(edges[x, y], edge_y[x,y]) 

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


def task(img):
    # 2. Convert to Grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. Apply filters

    for apply_filter in [box_blur, edge_detection, sharpen]:
        filtered_gray_img = grayscale_img.copy()
        name, filtered_gray_img = apply_filter(filtered_gray_img)
        
        # 4. Save each output
        cv2.imwrite(f'{name}.jpg', filtered_gray_img)

        # 5. prints image shape
        print(f'{name}\'s shape: ',  filtered_gray_img.shape)


'''
Note: Change load image URL to work.
'''
if __name__ == '__main__':
    # 1. Load
    img = load()
    task(img)