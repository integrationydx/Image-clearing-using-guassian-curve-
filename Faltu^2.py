import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import urllib.request

# ==========================================
# Member 1: Data Lead
# ==========================================
def load_and_add_noise(image_path="einstein.jpg"):
    """Loads an image, converts to grayscale uint8, and adds random noise."""
    # Attempt to load local image directly using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Could not find image at: {image_path}. Generating synthetic image...")
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
        cv2.circle(img, (128, 128), 50, 100, -1)

    # Ensure it's uint8
    img = img.astype(np.uint8)

    # Add random fluctuations (Gaussian Noise)
    row, col = img.shape
    mean = 0
    sigma = 30 # Noise intensity
    gauss_noise = np.random.normal(mean, sigma, (row, col))
    
    noisy_img = img + gauss_noise
    # Clip values to stay within valid pixel range 0-255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return img, noisy_img
# ==========================================
# Member 2: Simplification
# ==========================================
def apply_convolution_filter(noisy_img):
    """Applies a Gaussian blur matrix to clean the noisy image."""
    # Define a 5x5 Gaussian approximation kernel
    kernel_blur = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ]) / 273.0

    # Perform matrix convolution
    cleaned_img = convolve2d(noisy_img, kernel_blur, mode='same', boundary='symm')
    
    return np.clip(cleaned_img, 0, 255).astype(np.uint8)

# ==========================================
# Member 3: Math Analyst
# ==========================================
def calculate_residual(noisy_img, cleaned_img):
    """Calculates the mathematical difference between noisy and cleaned matrices."""
    # Convert to int32 to prevent underflow/overflow during subtraction
    residual_matrix = noisy_img.astype(np.int32) - cleaned_img.astype(np.int32)
    
    # To visualize the residual, we shift it by 128 so negative differences are visible
    residual_display = np.clip(residual_matrix + 128, 0, 255).astype(np.uint8)
    
    return residual_matrix, residual_display

# ==========================================
# Member 4: Application
# ==========================================
def apply_sharpening_and_plot(original, noisy, cleaned, residual_display):
    """Applies a sharpening filter and generates the final demo plots."""
    
    # Define sharpening kernel (Kernel_Sharp1)
    kernel_sharp = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])

    # Convolve cleaned image with sharpening kernel
    sharpened_img = convolve2d(cleaned, kernel_sharp, mode='same', boundary='symm')
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

    # Prepare demo plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    images = [original, noisy, cleaned, residual_display, sharpened_img]
    titles = ["1. Original Image", "2. Noisy Image", "3. Cleaned (Blurred)", "4. Residual (Noise Extracted)", "5. Final Sharpened"]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution Workflow
# ==========================================
# ==========================================
# Main Execution Workflow
# ==========================================
def main():
    print("Executing Linear Algebra Image Processing Pipeline...")
    
    # Step 1: Pass your local file path right here!
    my_image_path = r"C:\Users\aadis\OneDrive\Desktop\faltu.jpeg"
    original_img, noisy_img = load_and_add_noise(my_image_path)
    
    # Step 2
    cleaned_img = apply_convolution_filter(noisy_img)
    
    # Step 3
    raw_residual, residual_vis = calculate_residual(noisy_img, cleaned_img)
    
    # Using raw_residual to prove the filter's effectiveness
    average_noise_removed = np.mean(np.abs(raw_residual))
    max_noise_spike = np.max(np.abs(raw_residual))
    
    print(f"Filter Stats:")
    print(f"- Average noise removed per pixel: {average_noise_removed:.2f} units")
    print(f"- Largest single noise spike removed: {max_noise_spike} units")
    
    # Step 4
    apply_sharpening_and_plot(original_img, noisy_img, cleaned_img, residual_vis)

if __name__ == "__main__":
    main()