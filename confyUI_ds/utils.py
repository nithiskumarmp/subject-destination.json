import cv2
import numpy as np
import torch


def tensor_to_cv2(tensor_image):
    """Convert ComfyUI tensor to OpenCV format"""
    # ComfyUI images are (batch, height, width, channels) in RGB
    # OpenCV expects (height, width, channels) in BGR
    if len(tensor_image.shape) == 4:
        # Take first image from batch
        tensor_image = tensor_image[0]
    
    # Convert from tensor to numpy array (0-1 range to 0-255)
    cv2_image = tensor_image.cpu().numpy()
    cv2_image = (cv2_image * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    
    return cv2_image


def cv2_to_tensor(cv2_image):
    """Convert OpenCV image to ComfyUI tensor format"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to 0-1 range
    tensor_image = rgb_image.astype(np.float32) / 255.0
    
    # Convert to torch tensor and add batch dimension
    tensor_image = torch.from_numpy(tensor_image).unsqueeze(0)
    
    return tensor_image


def reorder(vertices):
    """Reorder vertices to top-left, top-right, bottom-right, bottom-left"""
    reordered = np.zeros_like(vertices, dtype=np.float32)
    add = vertices.sum(1)
    reordered[0] = vertices[np.argmin(add)]  # top-left (smallest sum)
    reordered[2] = vertices[np.argmax(add)]  # bottom-right (largest sum)
    diff = np.diff(vertices, axis=1)
    reordered[1] = vertices[np.argmin(diff)]  # top-right (smallest diff)
    reordered[3] = vertices[np.argmax(diff)]  # bottom-left (largest diff)
    return reordered


def blank_page(im):
    """Remove text using morphological operations and GrabCut"""
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=3)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (20, 20, img.shape[1]-20, img.shape[0]-20)
    
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img = img * mask2[:,:,np.newaxis]
    except:
        # Fallback: return original if GrabCut fails
        pass
    
    return img


def to_grayscale(im):
    """Convert image to grayscale"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def blur(im, kernel_size=5):
    """Apply bilateral filter for edge-preserving smoothing"""
    return cv2.bilateralFilter(im, kernel_size, 60, 60)


def to_edges(im, low_threshold=20, high_threshold=70):
    """Apply Canny edge detection"""
    return cv2.Canny(im, low_threshold, high_threshold)


def find_vertices(im):
    """Find document vertices using contour detection"""
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = im.shape
    area = (w - 10) * (h - 10)
    area_found = area * 0.5
    
    # Default fallback vertices (full image)
    contour = np.array([[1, 1], [1, h-1], [w-1, h-1], [w-1, 1]])
    
    for i in contours:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.03 * perimeter, True)
        
        if (len(approx) == 4 and 
            cv2.isContourConvex(approx) and 
            area_found < cv2.contourArea(approx) < area):
            area_found = cv2.contourArea(approx)
            contour = approx
            
    return contour.reshape((4, 2))


def crop_out(im, vertices):
    """Apply perspective transform to crop document"""
    vertices = reorder(vertices)
    (a, b, c, d) = vertices
    
    # Calculate optimal output dimensions
    w1 = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
    w2 = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
    width = max(int(w1), int(w2))
    
    h1 = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2))
    h2 = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2))
    height = max(int(h1), int(h2))
    
    # Define target rectangle
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], 
                     dtype="float32")
    
    try:
        transform = cv2.getPerspectiveTransform(vertices, target)
        return cv2.warpPerspective(im, transform, (width, height))
    except:
        # Fallback: return original image if perspective transform fails
        return im


def enhance_sharpening(im):
    """Apply sharpening enhancement"""
    kernel_sharpening = np.array([[0,-1,0], 
                                 [-1, 5,-1],
                                 [0,-1,0]])
    sharpened = cv2.filter2D(im, -1, kernel_sharpening)
    
    value1, value2 = 30, 25
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value1
    v[v > lim] = 255
    v[v <= lim] += value1
    
    lim = 255 - value2
    s[s > lim] = 255
    s[s <= lim] += value2
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def enhance_cartooning(im):
    """Apply cartooning enhancement"""
    num_down = 2
    num_bilateral = 7
    
    img_color = im
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
        
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    
    img_color = img_color[:im.shape[0], :im.shape[1], :im.shape[2]]

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, blockSize=15, C=3)
    
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_color, img_edge)


def enhance_clahe(im):
    """Apply CLAHE enhancement"""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def enhance_threshold(im):
    """Apply thresholding with denoising"""
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, 11, 31, 9)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def enhance_adaptive_threshold(im):
    """Apply adaptive thresholding with denoising"""
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 23, 5)
    denoised = cv2.fastNlMeansDenoising(thresh, 11, 31, 9)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def enhance_flat_field(im):
    """Apply flat field correction"""
    F = cv2.GaussianBlur(im, (401, 401), 0)
    C = np.int64(np.round((im * np.mean(F)) / F))
    C = np.clip(C, 0, 255).astype(np.uint8)
    return enhance_clahe(C)


def enhance_image(im, method="sharpening"):
    """Apply enhancement based on selected method"""
    enhancement_methods = {
        "sharpening": enhance_sharpening,
        "cartooning": enhance_cartooning,
        "clahe": enhance_clahe,
        "threshold": enhance_threshold,
        "adaptive_threshold": enhance_adaptive_threshold,
        "flat_field": enhance_flat_field
    }
    
    if method in enhancement_methods:
        return enhancement_methods[method](im)
    else:
        return enhance_sharpening(im)  # Default fallback