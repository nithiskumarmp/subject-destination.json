import cv2
import numpy as np
import torch
from .utils import tensor_to_cv2, cv2_to_tensor, reorder, crop_out, enhance_image


def detect_object_on_black_background(image, threshold=30):
    """Detect object boundaries against black background"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create binary mask: black background vs object
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary_mask


def find_object_contour_simple(binary_mask):
    """Find main object contour from binary mask"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: return full image bounds
        h, w = binary_mask.shape
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    
    # Get largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Try to approximate to quadrilateral
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Start with loose approximation and tighten if needed
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        approx = cv2.approxPolyDP(largest_contour, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape((4, 2))
    
    # If quad approximation fails, use minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def scan_black_background_object(image, enhancement_method="clahe", bg_threshold=30):
    """
    Optimized scanning for objects on black background
    
    Args:
        image: Input image (OpenCV format)
        enhancement_method: Enhancement to apply
        bg_threshold: Threshold to separate object from black background
    """
    try:
        # Step 1: Detect object against black background
        binary_mask = detect_object_on_black_background(image, bg_threshold)
        
        # Step 2: Find object contour
        vertices = find_object_contour_simple(binary_mask)
        
        # Step 3: Perspective correction
        cropped = crop_out(image, vertices)
        
        # Step 4: Enhancement (preserve patterns/textures)
        enhanced = enhance_image(cropped, enhancement_method)
        
        return enhanced, binary_mask
        
    except Exception as e:
        print(f"Black background scanning error: {str(e)}")
        return image, np.zeros_like(image[:,:,0])


class BlackBackgroundScannerNode:
    """
    Optimized ComfyUI node for objects on black backgrounds
    """
    
    CATEGORY = "image/processing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement": (["clahe", "sharpening", "flat_field", "none"], {
                    "default": "clahe"
                }),
                "background_threshold": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 5
                }),
                "return_mask": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("scanned_image", "detection_mask")
    FUNCTION = "scan_black_background"
    
    def scan_black_background(self, image, enhancement, background_threshold, return_mask):
        """
        Main function for black background object scanning
        """
        try:
            results = []
            masks = []
            
            # Process each image in batch
            for i in range(image.shape[0]):
                # Convert to OpenCV format
                cv2_image = tensor_to_cv2(image[i:i+1])
                
                # Process with optimized algorithm
                processed, mask = scan_black_background_object(
                    cv2_image, enhancement, background_threshold
                )
                
                # Convert back to tensors
                result_tensor = cv2_to_tensor(processed)
                results.append(result_tensor)
                
                if return_mask:
                    # Convert mask to 3-channel for visualization
                    mask_3ch = np.stack([mask, mask, mask], axis=2)
                    mask_tensor = cv2_to_tensor(mask_3ch)
                    masks.append(mask_tensor)
            
            # Combine results
            final_result = torch.cat(results, dim=0)
            
            if return_mask and masks:
                mask_result = torch.cat(masks, dim=0)
            else:
                mask_result = torch.zeros_like(final_result)
            
            return (final_result, mask_result)
            
        except Exception as e:
            print(f"BlackBackgroundScanner error: {str(e)}")
            empty_mask = torch.zeros_like(image)
            return (image, empty_mask)