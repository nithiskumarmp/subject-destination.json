import torch
import numpy as np
from .utils import (
    tensor_to_cv2, cv2_to_tensor, blank_page, to_grayscale, blur, 
    to_edges, find_vertices, crop_out, enhance_image
)


class DocumentScannerNode:
    """
    ComfyUI node for document scanning with perspective correction and enhancement
    """
    
    CATEGORY = "image/processing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_method": (["sharpening", "cartooning", "clahe", "threshold", "adaptive_threshold", "flat_field"], {
                    "default": "sharpening"
                }),
                "edge_threshold_low": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "edge_threshold_high": ("INT", {
                    "default": 70,
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "blur_kernel_size": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 15,
                    "step": 2
                }),
                "skip_preprocessing": ("BOOLEAN", {
                    "default": False
                }),
                "return_debug_edges": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("scanned_image", "debug_edges")
    FUNCTION = "scan_document"
    
    def scan_document(self, image, enhancement_method, edge_threshold_low, edge_threshold_high, 
                     blur_kernel_size, skip_preprocessing, return_debug_edges):
        """
        Main document scanning function
        """
        try:
            results = []
            debug_edges_batch = []
            
            # Process each image in the batch
            for i in range(image.shape[0]):
                # Convert tensor to OpenCV format
                cv2_image = tensor_to_cv2(image[i:i+1])
                
                # Document scanning pipeline
                processed_image, edges_debug = self._process_single_image(
                    cv2_image, enhancement_method, edge_threshold_low, 
                    edge_threshold_high, blur_kernel_size, skip_preprocessing
                )
                
                # Convert back to tensor format
                result_tensor = cv2_to_tensor(processed_image)
                results.append(result_tensor)
                
                if return_debug_edges:
                    edges_tensor = cv2_to_tensor(edges_debug)
                    debug_edges_batch.append(edges_tensor)
            
            # Combine batch results
            final_result = torch.cat(results, dim=0)
            
            if return_debug_edges and debug_edges_batch:
                debug_result = torch.cat(debug_edges_batch, dim=0)
            else:
                # Return empty tensor with same batch size if no debug requested
                debug_result = torch.zeros_like(final_result)
            
            return (final_result, debug_result)
            
        except Exception as e:
            print(f"DocumentScanner error: {str(e)}")
            # Fallback: return original image
            empty_debug = torch.zeros_like(image)
            return (image, empty_debug)
    
    def _process_single_image(self, cv2_image, enhancement_method, edge_threshold_low, 
                            edge_threshold_high, blur_kernel_size, skip_preprocessing):
        """
        Process a single image through the document scanning pipeline
        """
        try:
            original_image = cv2_image.copy()
            
            # Step 1: Preprocessing (optional)
            if not skip_preprocessing:
                processed_image = blank_page(cv2_image)
            else:
                processed_image = cv2_image
            
            # Step 2: Convert to grayscale
            grayscale = to_grayscale(processed_image)
            
            # Step 3: Apply blur
            blurred = blur(grayscale, blur_kernel_size)
            
            # Step 4: Edge detection
            edges = to_edges(blurred, edge_threshold_low, edge_threshold_high)
            
            # Create debug visualization of edges
            edges_debug = np.stack([edges, edges, edges], axis=2)  # Convert to 3-channel for visualization
            
            # Step 5: Find document vertices
            vertices = find_vertices(edges)
            
            # Step 6: Perspective correction
            cropped = crop_out(original_image, vertices)
            
            # Step 7: Enhancement
            enhanced = enhance_image(cropped, enhancement_method)
            
            return enhanced, edges_debug
            
        except Exception as e:
            print(f"Error processing single image: {str(e)}")
            # Fallback: return original image
            edges_debug = np.zeros_like(cv2_image)
            return cv2_image, edges_debug
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        This method can be used to determine if the node should be re-executed
        For now, we'll always execute to ensure fresh results
        """
        return float("NaN")


# Alternative node with simplified interface
class SimpleDocumentScannerNode:
    """
    Simplified version of DocumentScanner with fewer parameters
    """
    
    CATEGORY = "image/processing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement": (["auto", "text", "photo"], {
                    "default": "auto"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scanned_image",)
    FUNCTION = "simple_scan"
    
    def simple_scan(self, image, enhancement):
        """
        Simplified document scanning with preset configurations
        """
        # Map simple enhancement options to detailed methods
        enhancement_map = {
            "auto": "sharpening",
            "text": "adaptive_threshold", 
            "photo": "clahe"
        }
        
        enhancement_method = enhancement_map.get(enhancement, "sharpening")
        
        # Use DocumentScannerNode with default parameters
        scanner = DocumentScannerNode()
        result, _ = scanner.scan_document(
            image=image,
            enhancement_method=enhancement_method,
            edge_threshold_low=20,
            edge_threshold_high=70,
            blur_kernel_size=5,
            skip_preprocessing=False,
            return_debug_edges=False
        )
        
        return (result,)