import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_mask_from_boundary(src_image, boundary):
    '''
    Generate a mask on a source image based on 1D boundary
    
    Args :
        src_image (np.array) : source image
        boundary (np.array) : 1D boundary
        
    Returns:
        np.array result mask
    '''
    
    img_height = src_image.shape[0]
    img_width = src_image.shape[1]
    
    mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    
    for index, height_y in enumerate(boundary):
        mask[:height_y, index] = 255
        
    return mask

def convert_img_to_gray(src_image):
    '''
    Simple method to convert a source image to gray
    
    Args :
        src_image (np.array) : source image
        
    Returns :
        np.array gray image
    '''
    
    return cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

def convert_image_to_gradient(src_image):
    '''
    This method computes Sobel operators in horizontal and vertical directions and combine both. 
    Sobel operator is an aproximation of the gradient of an image.
    For more information on Sobel operators : [[https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html]]
    
    The method first ensures the image is gray
    
    Args :
        src_image (np.array) : source image
        
    Returns :
        np.array gray image
    '''
    
    gray_image = convert_img_to_gray(src_image)
    
    return np.hypot(
        cv2.Sobel(gray_image, cv2.CV_64F, 1, 0),
        cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    )
    
def compute_energy_fuction(boundary_tmp, src_image):
    '''
    Define an energy function to be optimized.
    This method takes a 1D boundary array and a source image to compute its energy function output.
    
    Args :
        boundary_tmp (np.array) : 1D boundary
        src_image (np.array): Source Image
        
    Returns :
        float Energy function output
    '''
    
    current_sky_mask = generate_mask_from_boundary(src_image, boundary_tmp)
    
    ground_array = np.ma.array(
        src_image,
        mask=cv2.cvtColor(cv2.bitwise_not(current_sky_mask), cv2.COLOR_GRAY2BGR)
    ).compressed()
    
    sky_array = np.ma.array(
        src_image,
        mask=cv2.cvtColor(current_sky_mask, cv2.COLOR_GRAY2BGR)
    ).compressed()
    
    ground_array.shape = (ground_array.size//3, 3)
    sky_array.shape = (sky_array.size//3, 3)

    sigma_g, mu_g = cv2.calcCovarMatrix(
        ground_array,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )
    
    sigma_s, mu_s = cv2.calcCovarMatrix(
        sky_array,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )

    y = 2

    return 1 / (
        (y * np.linalg.det(sigma_s) + np.linalg.det(sigma_g)) +
        (y * np.linalg.det(np.linalg.eig(sigma_s)[1]) +
        np.linalg.det(np.linalg.eig(sigma_g)[1])))

def compute_boundary(gradient_image, threshold):
    '''
    This methods compute the horizontal boundary from the gradient image given a min threshold
    
    Args : 
        gradient_image (np.array): Gradient Image
        threshold (int) : Threshold 
        
    Returns :
        np.array 1D boundary
    '''
    
    img_height = gradient_image.shape[0]
    img_width = gradient_image.shape[1]
    
    # Initialize 1D array with a boundary at the top of the image (image height)
    boundary = np.full(img_width, img_height)
    
    for x in range(img_width):
        
        # Searching for the higher gradient index above the threshold 
        border_pos = np.argmax(gradient_image[:, x] > threshold)

        # argmax returns 0 if nothing is > threshold
        if border_pos > 0:
            boundary[x] = border_pos
            
    return boundary

def compute_optimal_boundary(src_image, min_gradient_threshold=5, max_gradient_threshold=600, step=5):
    '''
    Iterative process to compute a boundary and optimize its energy function output
    
    Args : 
        src_image (np.array) : Source Image
        min_gradient_threshold (int) : Min gradient threshold to look at
        max_gradient_threshold (int) : Max gradient threshold to look at
        step (int) : Steps between min and max threshold
    
    Returns :
        np.array 1D optimal boundary
    '''
        
    # Compute the gradient from the image
    gradient_image = convert_image_to_gradient(src_image)

    number_of_steps = ((max_gradient_threshold - min_gradient_threshold) // step) + 1
    
    optimal_boundary = None
    jn_max = 0

    for k in range(1, number_of_steps + 1):
        
        threshold = min_gradient_threshold + ((max_gradient_threshold - min_gradient_threshold) // number_of_steps - 1) * (k - 1)

        boundary_tmp = compute_boundary(gradient_image, threshold)
        jn = compute_energy_fuction(boundary_tmp, src_image)

        if jn > jn_max:
            jn_max = jn
            optimal_boundary = boundary_tmp

    return optimal_boundary

def get_detected_sky_mask(src_image):
    '''
    This methods takes an image, computes its optimal boundary and generates 
    a mask with the sky delimitation from the rest

    Args :
        src_image (np.array) : source image
        
    Returns :
        np.array sky delimitation mask
    '''

    optimal_boundary = compute_optimal_boundary(src_image)
    
    return generate_mask_from_boundary(src_image, optimal_boundary)