import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch
import io
from PIL import Image

class SeamlessPatternExtractor:
    def __init__(self, image_input):
        # image_input is numpy (H, W, C) BGR (or grayscale if adapted, but we pass BGR)
        self.original = image_input
        self.orig_h, self.orig_w = self.original.shape[:2]
        
        if self.original is None:
            raise ValueError("Image is None")
        
        # Ensure we have data
        if self.original.size == 0:
             raise ValueError("Image is empty")
             
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.leveled = None
        self.tile_w = 0
        self.tile_h = 0

    def run(self):
        # print("--- 1. Detecting Orientation & Width ---")
        angle, self.tile_w = self._get_orientation_and_width()
        
        # print("--- 2. Leveling Image (Horizontal Fix) ---")
        self.leveled = self._safe_rotate(self.original, angle)
        
        # print("--- 3. Detecting Height & Vertical Drift ---")
        h, w, _ = self.leveled.shape
        # Use a center strip to avoid edge artifacts
        strip = self.leveled[:, max(0, w//2 - self.tile_w): min(w, w//2 + self.tile_w)] 
        
        # We now get height AND drift (shift_x)
        self.tile_h, drift_x = self._find_height_and_drift(strip)
        
        # print("--- 4. Correcting Vertical Tilt (Shear) ---")
        # If there is significant drift, shear the image
        if abs(drift_x) > 2: # Tolerance threshold
            self.leveled = self._apply_vertical_shear(self.leveled, drift_x, self.tile_h)
            # Update gray for the grid search step
            self.gray = cv2.cvtColor(self.leveled, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = cv2.cvtColor(self.leveled, cv2.COLOR_BGR2GRAY)
            
        # print("--- 5. Finding Best Seamless Crop (Grid Search) ---")
        best_tile, heatmap = self._find_best_starting_point()
        
        debug_img = self._visualize_results(heatmap, best_tile)
        
        final_h, final_w = best_tile.shape[:2]
        
        # Calculate Ratios
        # Avoid division by zero
        width_ratio = self.orig_w / final_w if final_w > 0 else 1.0
        height_ratio = self.orig_h / final_h if final_h > 0 else 1.0

        return best_tile, final_w, final_h, width_ratio, height_ratio, debug_img

    def _get_orientation_and_width(self):
        # A. Calculate the Mean Color
        mean_color = int(np.mean(self.gray))

        # B. Split Image
        h, w = self.gray.shape
        
        # --- Pass 1: Rough Angle ---
        template = self.gray[:, 0:w//2]
        t_h, t_w = template.shape

        # C. Pad the Source vertically
        pad_v = int(h*0.07)
        source = cv2.copyMakeBorder(self.gray, pad_v, pad_v, 0, w//2, cv2.BORDER_CONSTANT, value=mean_color)
        
        # D. Match
        res = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        # Mask out the immediate overlap (we want the *next* repetition)
        res[:, 0:int(w*0.1)] = 0 
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        # Calculate Angle
        dy = max_loc[1] - pad_v
        dx = max_loc[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        # --- Pass 2: Exact Width on Leveled Temp ---
        temp_leveled = self._safe_rotate(self.gray, angle)
        h_l, w_l = temp_leveled.shape
        
        # Pad right only (to find horizontal repeat)
        source_l = cv2.copyMakeBorder(temp_leveled, pad_v, pad_v, 0, w//2, cv2.BORDER_CONSTANT, value=mean_color)
        template_l = temp_leveled[:, 0:w_l//2]
        t_h_l, t_w_l = template_l.shape
        
        res_l = cv2.matchTemplate(source_l, template_l, cv2.TM_CCOEFF_NORMED)
        res_l[:, 0:int(w_l*0.1)] = 0
        _, _, _, max_loc_l = cv2.minMaxLoc(res_l)

        return angle, max_loc_l[0]

    def _find_height_and_drift(self, strip_img):
        gray_strip = cv2.cvtColor(strip_img, cv2.COLOR_BGR2GRAY)
        h, w = gray_strip.shape
        mean_color = int(np.mean(gray_strip))

        # 1. Template is top half
        template = gray_strip[0:h//2, :]
        t_h, t_w = template.shape

        # 2. Pad Source heavily on all sides to catch the drift
        pad_v = h // 2
        pad_h = int(h*0.07) # Pad width significantly to find the drifted match
        source = cv2.copyMakeBorder(gray_strip, pad_v, pad_v, pad_h, pad_h, cv2.BORDER_CONSTANT, value=mean_color)
        
        # 3. Match
        res = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        
        # 4. Mask out self-match (center area)
        res[0:int(pad_v + h*0.1), :] = 0
        
        # 5. Find match
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        # 6. Calculate Metrics
        detected_height = max_loc[1] - pad_v
        drift_x = max_loc[0] - pad_h
        
        return detected_height, drift_x

    def _apply_vertical_shear(self, img, drift_x, height_y):
        h, w = img.shape[:2]
        shear_factor = -drift_x / height_y
        M = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])
        
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        new_corners = cv2.transform(np.array([corners]), M)[0]
        
        x_min = new_corners[:, 0].min()
        x_max = new_corners[:, 0].max()
        new_w = int(x_max - x_min)
        
        M[0, 2] = -x_min
        
        sheared = cv2.warpAffine(img, M, (new_w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return sheared

    def _safe_rotate(self, img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        angle_rad = math.radians(abs(angle))
        y_margin = int(w * math.sin(angle_rad)) + 5
        x_margin = int(h * math.sin(angle_rad)) + 5
        
        if y_margin*2 < h and x_margin*2 < w:
            return rotated[y_margin:h-y_margin, x_margin:w-x_margin]
        return rotated

    def _find_best_starting_point(self):
        img = self.gray
        img_h, img_w = img.shape
        t_h, t_w = self.tile_h, self.tile_w
        
        valid_h = img_h - t_h
        valid_w = img_w - t_w
        
        if valid_h <= 0 or valid_w <= 0:
            t_h = min(t_h, img_h)
            t_w = min(t_w, img_w)
            # Create a "fake" heatmap for visualization
            return self.leveled[0:t_h, 0:t_w], np.zeros((1,1), dtype=np.float32)

        # 1. Vertical Seam Cost
        diff_v = cv2.absdiff(img[0:valid_h, :], img[t_h:, :])
        cost_v = cv2.boxFilter(diff_v.astype(np.float32), -1, (t_w, 1), normalize=False)
        cost_v = cost_v[:, 0:valid_w]

        # 2. Horizontal Seam Cost
        diff_h = cv2.absdiff(img[:, 0:valid_w], img[:, t_w:]) 
        cost_h = cv2.boxFilter(diff_h.astype(np.float32), -1, (1, t_h), normalize=False)
        cost_h = cost_h[0:valid_h, :]

        # 3. Total Cost Map
        total_cost = cost_v + cost_h
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(total_cost)
        best_x, best_y = min_loc
        
        best_tile = self.leveled[best_y:best_y+t_h, best_x:best_x+t_w]
        
        return best_tile, total_cost

    def _visualize_results(self, cost_map, best_tile):
        # Create a figure and save to numpy array
        fig = plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 3, 1)
        # Normalize cost map if it has content
        if cost_map.size > 1:
            display_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX)
            plt.imshow(display_map, cmap='jet')
        else:
            plt.text(0.5, 0.5, "No Heatmap", ha='center')
        plt.title("Seam Error Heatmap\n(Blue = Good, Red = Bad)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        # best_tile is BGR
        plt.imshow(cv2.cvtColor(best_tile, cv2.COLOR_BGR2RGB))
        plt.title(f"Selected Tile ({best_tile.shape[1]}x{best_tile.shape[0]})")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        row = np.concatenate([best_tile, best_tile, best_tile], axis=1)
        grid = np.concatenate([row, row, row], axis=0)
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        plt.title("2x2 Verification (actually 3x3)")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # Load from buffer
        img_pil = Image.open(buf)
        img_np = np.array(img_pil)
        
        if img_np.shape[2] == 4:
            return img_np[:,:,:3] 
        else:
            return img_np

class SeamlessPatternNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("best_tile", "debug_preview", "tile_width", "tile_height", "width_ratio", "height_ratio")
    FUNCTION = "run"
    CATEGORY = "Image Processing"

    def run(self, image):
        # image is [B, H, W, C] in RGB, float 0-1
        results_tiles = []
        results_debugs = []
        widths = []
        heights = []
        w_ratios = []
        h_ratios = []

        # Process batch
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            extractor = SeamlessPatternExtractor(img_bgr)
            try:
                best_tile, w, h, w_ratio, h_ratio, debug_img_rgb = extractor.run()
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                raise e
            
            # best_tile is BGR uint8
            tile_rgb = cv2.cvtColor(best_tile, cv2.COLOR_BGR2RGB)
            tile_tensor = torch.from_numpy(tile_rgb).float() / 255.0
            results_tiles.append(tile_tensor)
            
            # debug_img_rgb is RGB int (from PIL)
            debug_tensor = torch.from_numpy(debug_img_rgb).float() / 255.0
            results_debugs.append(debug_tensor)

            widths.append(w)
            heights.append(h)
            w_ratios.append(w_ratio)
            h_ratios.append(h_ratio)

        if not results_tiles:
             return (torch.empty(0), torch.empty(0), 0, 0, 0.0, 0.0)
             
        try:
            final_tiles = torch.stack(results_tiles)
            final_debugs = torch.stack(results_debugs)
        except RuntimeError:
            print("Warning: Batch output sizes differ. Returning only the first result.")
            final_tiles = results_tiles[0].unsqueeze(0)
            final_debugs = results_debugs[0].unsqueeze(0)
        
        return (final_tiles, final_debugs, widths[0], heights[0], w_ratios[0], h_ratios[0])
