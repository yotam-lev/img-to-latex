# Install required packages
#!pip install scikit-image opencv-python-headless

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Image processing libraries
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
#from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, rectangle, erosion, thin
from skimage.util import img_as_ubyte

# Additional libraries
import cv2
#from google.colab import files

def debug_image(image, step_name):
    """
    Prints debug information about an image.

    Parameters:
    - image: The NumPy array representing the image.
    - step_name: A string indicating which step in the pipeline the image corresponds to.
    """
    print(f"Step: {step_name}")
    print(f" - Shape: {image.shape}")
    print(f" - Data Type: {image.dtype}")
    print(f" - Min: {image.min()}, Max: {image.max()}")
    print(f" - Mean: {np.mean(image):.4f}")
    print(f" - Unique Values: {np.unique(image)}\n")


def pad_image(img, show_image=False, fig=None, axes=None):
    """
    Add 10% white padding to all sides of an input image.

    Parameters:
    img (numpy.ndarray): Input image as a NumPy array.
    show_image (bool, optional): If True, displays the padded image.
    fig (matplotlib.figure.Figure, optional): Matplotlib figure for unified plotting.
    axes (matplotlib.axes.Axes, optional): Axis to use for displaying the image.

    Returns:
    numpy.ndarray: The image with 10% white padding on all sides.
    """
    h, w = img.shape[:2]
    pad_top = pad_bottom = int(0.1 * h)
    pad_left = pad_right = int(0.1 * w)

    if len(img.shape) == 3:
        padded_img = np.pad(
            img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant", constant_values=255
        )
    else:
        padded_img = np.pad(
            img, ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant", constant_values=255
        )

    if show_image:
        if fig is not None and axes is not None:
            axes.imshow(padded_img, cmap="gray" if len(img.shape) == 2 else None)
            axes.set_title("Padded Image")
            axes.axis("off")
        else:
            plt.imshow(padded_img, cmap="gray" if len(img.shape) == 2 else None)
            plt.title("Padded Image")
            plt.axis("off")
            plt.show()

    return padded_img


def pad_image(img, show_image=False):
    """
    Add 10% white padding to all sides of an input image.

    Parameters:
    -----------
    img : np.ndarray
        Input image as a NumPy array. Can be grayscale (H x W) or color (H x W x C).
    show_image : bool, optional
        If True, displays the padded image using matplotlib (default=False).

    Returns:
    --------
    np.ndarray
        The image with 10% white padding on all sides.
    """
    # Calculate padding amounts (10% of the height and width)
    h, w = img.shape[:2]
    pad_top = pad_bottom = int(0.1 * h)
    pad_left = pad_right = int(0.1 * w)

    # Apply padding based on whether the image is grayscale or color
    if len(img.shape) == 3:  # Color image (H x W x C)
        padded_img = np.pad(
            img,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=255  # White padding for RGB
        )
    else:  # Grayscale image (H x W)
        padded_img = np.pad(
            img,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=255  # White padding for grayscale
        )

    # Optionally display the padded image
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(padded_img, cmap='gray' if len(img.shape) == 2 else None)
        plt.axis('off')
        plt.title('Padded Image')
        plt.show()

    return padded_img

def process_image(image, target_size=45, show_image=False, show_process=False):
    """
    Process the input image through multiple steps: padding, binarization, erosion, inversion,
    thinning, shifting, scaling, and centering.

    Steps:
    1. Pad the image.
    2. Binarize the image.
    3. Apply erosion.
    4. Invert the image.
    5. Thin the image.
    6. Shift the image to align content to the top-left.
    7. Scale the image to fit within a target size.
    8. Center the image in a square box.

    Parameters:
    -----------
    image : np.ndarray
        The input image (RGB, RGBA, or grayscale).
    target_size : int, optional
        The target size for scaling and centering (default=45).
    show_image : bool, optional
        Whether to show the final processed image (default=False).
    show_process : bool, optional
        Whether to display intermediate images at each step (default=False).

    Returns:
    --------
    np.ndarray
        The final processed image, centered in a square box.
    """
    def debug_image(img, title):
        """Helper function to display debug information."""
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Step 0: Pad the image
    padded_image = pad_image(image, show_image=show_process)
    if show_process:
        debug_image(padded_image, "0. Padded Image")

    # Step 1: Binarize the image
    binary_image = binarize(padded_image, show_image=show_process)
    if show_process:
        debug_image(binary_image, "1. Binarized Image")

    # Step 2: Apply erosion
    eroded_image = erode_image(binary_image, show_image=show_process)
    if show_process:
        debug_image(eroded_image, "2. Eroded Image")

    # Step 3: Invert the image
    inverted_image = invert_image(eroded_image, show_image=show_process)
    if show_process:
        debug_image(inverted_image, "3. Inverted Image")

    # Step 4: Thin the image
    thinned_image = thin_image(inverted_image, show_image=show_process)
    if show_process:
        debug_image(thinned_image, "4. Thinned Image")

    # Step 5: Shift the image
    shifted_image = shift_image(thinned_image, show_image=show_process)
    if show_process:
        debug_image(shifted_image, "5. Shifted Image")

    # Step 6: Scale the image
    scaled_image = scale_image(shifted_image, target_size=target_size, show_image=show_process)
    if show_process:
        debug_image(scaled_image, "6. Scaled Image")

    # Step 7: Center the image
    centered_image = center_image(scaled_image, target_size=target_size, show_image=show_process)
    if show_process:
        debug_image(centered_image, "7. Centered Image")

    # Optionally show only the final image
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(centered_image, cmap='gray')
        plt.title('Final Processed Image')
        plt.axis('off')
        plt.show()

    return centered_image

def binarize(image, show_image=False):
    """
    Convert an RGBA or RGB image to a binary image.

    Parameters:
    image (numpy.ndarray): Input image in RGBA or RGB format.
    show_image (bool): If True, displays the grayscale and binary images.

    Returns:
    numpy.ndarray: Binary image (grayscale with pixel values 0 or 255).
    """
    # If the image has 4 channels, drop the alpha channel
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    # Convert to grayscale
    grayscale_img = rgb2gray(image)

    # (Optional) Apply Gaussian Blur for noise reduction
    gaussian_blur = gaussian(grayscale_img, sigma=1)

    # Apply Otsu threshold
    thresh = threshold_otsu(gaussian_blur)

    # Convert to binary
    binary_img = gaussian_blur > thresh

    # Convert back to grayscale (0 or 255)
    binary_img = (binary_img * 255).astype(np.uint8)

    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(grayscale_img, cmap='gray')
        ax1.set_title('Grayscale')
        ax1.axis('off')
        ax2.imshow(binary_img, cmap='gray')
        ax2.set_title('Binarized (Otsu)')
        ax2.axis('off')
        plt.show()

    return binary_img

def erode_image(binary_img, show_image=False):
    """
    Apply erosion to reduce noise and refine binary image features.

    Parameters:
    binary_img (numpy.ndarray): Binary image to be eroded (pixel values 0 or 255).
    show_image (bool): If True, displays the original binary and eroded images.

    Returns:
    numpy.ndarray: Eroded binary image.
    """
    # Create a 3x3 rectangular footprint for the erosion process
    footprint = rectangle(3, 3)
    eroded_img = erosion(binary_img, footprint=footprint)

    # Display the original and eroded images if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(binary_img, cmap='gray')
        ax1.set_title('Original Binary Image')
        ax1.axis('off')
        ax2.imshow(eroded_img, cmap='gray')
        ax2.set_title('Eroded Image')
        ax2.axis('off')
        plt.show()

    return eroded_img

def invert_image(eroded_img, show_image=False):
    """
    Invert the colors of a binary or grayscale image (black to white and white to black).

    Parameters:
    eroded_img (numpy.ndarray): Eroded image to be inverted (pixel values 0-255).
    show_image (bool): If True, displays the original eroded and inverted images.

    Returns:
    numpy.ndarray: Image with inverted colors.
    """
    # Maximum intensity value for inversion
    max_intensity = 255

    # Perform color inversion
    inverted_img = max_intensity - eroded_img

    # Display the original and inverted images if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(eroded_img, cmap='gray')
        ax1.set_title('Original Eroded Image')
        ax1.axis('off')
        ax2.imshow(inverted_img, cmap='gray')
        ax2.set_title('Inverted Image')
        ax2.axis('off')
        plt.show()

    return inverted_img

def thin_image(inverted_img, show_image=False):
    """
    Apply a thinning algorithm to refine binary image structures.

    Parameters:
    inverted_img (numpy.ndarray): Binary inverted image with pixel values 0 or 255.
    show_image (bool): If True, displays the original inverted and thinned images.

    Returns:
    numpy.ndarray: Grayscale thinned image with pixel values 0 or 255.
    """
    # Ensure the input image is binary (values 0 and 1)
    if inverted_img.dtype != bool:
        binary_img = inverted_img > 0
    else:
        binary_img = inverted_img

    # Apply the thinning algorithm
    thinned_img = thin(binary_img)

    # Convert the thinned binary image to grayscale (values 0 and 255)
    thinned_img_uint8 = img_as_ubyte(thinned_img)

    # Display the original and thinned images if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(inverted_img, cmap='gray')
        ax1.set_title('Original Inverted Image')
        ax1.axis('off')
        ax2.imshow(thinned_img_uint8, cmap='gray')
        ax2.set_title('Thinned Image')
        ax2.axis('off')
        plt.show()

    return thinned_img_uint8


def shift_image(image, show_image=False):
    """
    Shift an image so that its content starts from the top-left corner.

    Parameters:
    image (numpy.ndarray): Input binary or grayscale image (expected to be uint8).
    show_image (bool): If True, displays the original and shifted images.

    Returns:
    numpy.ndarray: Shifted image cropped to contain only non-zero content, retaining the same data type.
    """
    # Ensure the input image is in the correct range and type
    if image.dtype != np.uint8:
        print("Converting input image to uint8 format.")
        image = (image * 255).astype(np.uint8)

    # Identify rows and columns containing non-zero pixels
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)

    # Find the bounding indices of the non-zero content
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        # No content found, return the original image
        if show_image:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(image, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            ax2.imshow(image, cmap='gray')
            ax2.set_title('Shifted Image (No Content Found)')
            ax2.axis('off')
            plt.show()
        return image

    # Crop the image to the bounds of the content
    y_min, y_max = row_indices[[0, -1]]
    x_min, x_max = col_indices[[0, -1]]
    shifted_img = image[y_min:y_max+1, x_min:x_max+1]

    # Ensure the shifted image retains the correct range and type
    if shifted_img.dtype != np.uint8:
        shifted_img = (shifted_img * 255).astype(np.uint8)

    # Display the original and shifted images if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(shifted_img, cmap='gray')
        ax2.set_title('Shifted Image')
        ax2.axis('off')
        plt.show()

    return shifted_img

def get_scale(width, height, target_size):
    """
    Calculate the scaling factor to resize dimensions while maintaining the aspect ratio 
    and ensuring points remain within the target size bounds.

    Parameters:
    width (int): The current width of the object or image.
    height (int): The current height of the object or image.
    target_size (int): The target size for scaling, typically the maximum allowed dimension.

    Returns:
    float: The scaling factor to resize the object or image.
    """
    # Guard against zero dimensions to prevent division errors
    if height == 0 or width == 0:
        return 1.0  # Default scaling factor (no resizing)

    # Calculate the aspect ratio
    ratio = width / height

    # Determine the scaling factor based on the aspect ratio
    if ratio < 1.0:  # Taller than wide
        return (target_size - 1) / height
    else:  # Wider than tall or square
        return (target_size - 1) / width


def scale_image(image, target_size=45, show_image=False):
    """
    Scale an image to fit within a target size while maintaining its aspect ratio.

    Parameters:
    image (numpy.ndarray): Input binary or grayscale image, ideally with dtype=uint8.
    target_size (int): Maximum size for the longest dimension of the scaled image.
    show_image (bool): If True, displays the original and scaled images side by side.

    Returns:
    numpy.ndarray: Scaled image with preserved aspect ratio.
    """
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        print("Converting input image to uint8.")
        image = (image * 255).astype(np.uint8)

    # Get the original dimensions of the image
    height, width = image.shape

    # Handle edge case for zero or very small dimensions
    if height == 0 or width == 0:
        print("Warning: Image has zero dimension, returning original.")
        return image

    # Calculate the scaling factor
    scale = get_scale(width, height, target_size)

    # Get coordinates of non-zero (black) pixels
    y, x = np.nonzero(image)
    points = np.column_stack((x, y))

    # Scale the pixel coordinates
    scaled_points = np.floor(points * scale).astype(np.int32)

    # Calculate new dimensions based on the scale
    new_width = int(np.floor(width * scale)) + 1
    new_height = int(np.floor(height * scale)) + 1

    # Create a new blank image with the scaled dimensions, initialized to white
    scaled_img = np.ones((new_height, new_width), dtype=np.uint8) * 255

    # Filter points to ensure they stay within the bounds of the new image
    mask = (scaled_points[:, 0] < new_width) & (scaled_points[:, 1] < new_height)
    valid_points = scaled_points[mask]

    # Set valid scaled points to black
    scaled_img[valid_points[:, 1], valid_points[:, 0]] = 0

    # Display the original and scaled images if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f'Original Image ({width}x{height})')
        ax1.axis('off')
        ax2.imshow(scaled_img, cmap='gray')
        ax2.set_title(f'Scaled Image ({new_width}x{new_height})')
        ax2.axis('off')
        plt.show()

    return scaled_img


def center_image(scaled_img, target_size=45, show_image=False):
    """
    Center a scaled image within a target_size x target_size square box.

    Parameters:
    scaled_img (numpy.ndarray): Input scaled image (expected to be uint8 with grayscale values).
    target_size (int): Size of the square output box (default: 45).
    show_image (bool): If True, displays the scaled and centered images.

    Returns:
    numpy.ndarray: Centered image within a square box, with pixel values 0 (black) and 255 (white).
    """
    # Ensure the input image is uint8
    if scaled_img.dtype != np.uint8:
        print("Converting scaled image to uint8 format.")
        scaled_img = (scaled_img * 255).astype(np.uint8)

    # Initialize a blank white canvas with the target size
    centered_img = np.ones((target_size, target_size), dtype=np.uint8) * 255

    # Get the dimensions of the scaled image
    height, width = scaled_img.shape

    # Calculate margins to center the scaled image
    margin_x = int((target_size - width) / 2)
    margin_y = int((target_size - height) / 2)

    # Get coordinates of black pixels in the scaled image
    y, x = np.nonzero(scaled_img == 0)
    points = np.column_stack((x, y))

    # Add margins to center the black pixels
    centered_points = np.add(points, [margin_x, margin_y])

    # Ensure the centered points are within the bounds of the target size
    valid_points = centered_points[
        (centered_points[:, 0] >= 0) & (centered_points[:, 0] < target_size) &
        (centered_points[:, 1] >= 0) & (centered_points[:, 1] < target_size)
    ]

    # Set the valid points to black on the white canvas
    centered_img[valid_points[:, 1], valid_points[:, 0]] = 0

    # Display the original scaled image and the centered image if requested
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(scaled_img, cmap='gray')
        ax1.set_title(f'Scaled Image ({width}x{height})')
        ax1.axis('off')
        ax2.imshow(centered_img, cmap='gray')
        ax2.set_title(f'Centered Image ({target_size}x{target_size})')
        ax2.axis('off')
        plt.show()

    return centered_img


def process_image(image, target_size=45, show_image=False, show_process=False):
    """
    Process the image through the following steps:
    1. Binarize
    2. Erode
    3. Invert
    4. Thin
    5. Shift
    6. Scale
    7. Center

    Parameters:
    -----------
    image : np.ndarray
        The input image (RGB, RGBA, or grayscale).
    target_size : int, optional
        The target size for scaling and centering (default=45).
    show_image : bool, optional
        Whether to show only the final processed image (default=False).
    show_process : bool, optional
        Whether to show intermediate images and debug info at each step (default=False).

    Returns:
    --------
    np.ndarray
        The final centered image.
    """

    # 1. Binarize image
    binary_image = binarize(image, show_image=show_process)
    if show_process:
        debug_image(binary_image, "1. Binarize")

    # 2. Apply erosion
    eroded_image = erode_image(binary_image, show_image=show_process)
    if show_process:
        debug_image(eroded_image, "2. Erode")

    # 3. Invert image
    inverted_image = invert_image(eroded_image, show_image=show_process)
    if show_process:
        debug_image(inverted_image, "3. Invert")

    # 4. Thin image
    thinned_image = thin_image(inverted_image, show_image=show_process)
    if show_process:
        debug_image(thinned_image, "4. Thin")

    # 5. Shift image
    shifted_image = shift_image(thinned_image, show_image=show_process)
    if show_process:
        debug_image(shifted_image, "5. Shift")

    # 6. Scale image
    scaled_image = scale_image(shifted_image, target_size=target_size, show_image=show_process)
    if show_process:
        debug_image(scaled_image, "6. Scale")

    # 7. Center image
    centered_image = center_image(scaled_image, target_size=target_size, show_image=show_process)
    if show_process:
        debug_image(centered_image, "7. Center")

    # Optionally show only the final image
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(centered_image, cmap='gray')
        plt.title('Final Processed Image')
        plt.axis('off')
        plt.show()

    return centered_image


