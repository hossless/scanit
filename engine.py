import os
import cv2
import torch
import numpy as np

from model import HeatmapCornerNet, UNetEnhancer, DirectRegressionNet

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_coords_from_heatmap(heatmaps):
    coords = []
    for i in range(4):
        y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        coords.append([x, y])
    return np.array(coords)


class DocumentScannerEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_corner_model_info(self, version_name):
        base_path = os.path.join("models", version_name)
        if os.path.exists(os.path.join(base_path, "direct_best.pth")):
            return "direct", os.path.join(base_path, "direct_best.pth")
        elif os.path.exists(os.path.join(base_path, "heatmap_best.pth")):
            return "heatmap", os.path.join(base_path, "heatmap_best.pth")
        return None, None
        
    def load_corner_model(self, version_name):
        model_type, model_path = self.get_corner_model_info(version_name)
        
        if not model_type:
            raise FileNotFoundError(f"Oops! No valid weights found in models/{version_name}/")
            
        if model_type == "direct":
            model = DirectRegressionNet().to(self.device)
        else:
            model = HeatmapCornerNet().to(self.device)
            
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model, model_type

    def load_enhancer_model(self, version_name):
        model_path = os.path.join("models", version_name, "enhancer_best.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Enhancer weights missing at {model_path}")
        model = UNetEnhancer().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def detect_corners(self, img_rgb, corner_model, model_type):
        orig_h, orig_w = img_rgb.shape[:2]
        corner_input = cv2.resize(img_rgb, (256, 256))
        corner_tensor = torch.tensor(corner_input, dtype=torch.float32).permute(2, 0, 1) / 255.0
        corner_tensor = corner_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = corner_model(corner_tensor)[0]
            
        if model_type == "direct":
            raw_coords = (predictions.view(4, 2) * 256).cpu().numpy()
        else:
            heatmaps = predictions.cpu().numpy()
            raw_coords = extract_coords_from_heatmap(heatmaps)
            
        scaled_coords = np.zeros_like(raw_coords, dtype=np.float32)
        scaled_coords[:, 0] = (raw_coords[:, 0] / 256.0) * orig_w
        scaled_coords[:, 1] = (raw_coords[:, 1] / 256.0) * orig_h
        
        return order_points(scaled_coords)

    def warp_image(self, img, ordered_corners):
        tl, tr, br, bl = ordered_corners

        width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_A), int(width_B))

        height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_A), int(height_B))

        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        return cv2.warpPerspective(img, transform_matrix, (max_width, max_height))

    def enhance_tiled(self, img_rgb, enhancer_model, patch_size=512, overlap=0.5):
        h, w, c = img_rgb.shape
        stride = int(patch_size * (1 - overlap))

        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        new_h, new_w, _ = img_padded.shape

        result_accumulator = np.zeros((new_h, new_w, c), dtype=np.float32)
        weight_accumulator = np.zeros((new_h, new_w, c), dtype=np.float32)

        y_grid, x_grid = np.ogrid[:patch_size, :patch_size]
        center = patch_size / 2
        dist = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
        patch_weight = np.clip(1.0 - (dist / (patch_size / 2 * 1.5)), 0.1, 1.0)
        patch_weight = np.expand_dims(patch_weight, axis=-1)

        for y in range(0, new_h - patch_size + 1, stride):
            for x in range(0, new_w - patch_size + 1, stride):
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1) / 255.0
                patch_tensor = patch_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred_patch = enhancer_model(patch_tensor)[0]

                pred_patch_np = pred_patch.cpu().permute(1, 2, 0).numpy()
                result_accumulator[y:y+patch_size, x:x+patch_size] += pred_patch_np * patch_weight
                weight_accumulator[y:y+patch_size, x:x+patch_size] += patch_weight

        final_padded_img = result_accumulator / weight_accumulator
        final_img = final_padded_img[:h, :w]
        
        final_img_uint8 = np.clip(final_img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(final_img_uint8, cv2.COLOR_RGB2BGR)

    def visualize_corners(self, image_path, corner_version):
        print(f"Drawing corner detection for {corner_version}...")
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        corner_model, model_type = self.load_corner_model(corner_version)
        
        corner_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        corner_tensor = corner_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = corner_model(corner_tensor)[0]

        if model_type == "direct":
            coords = (predictions.view(4, 2) * 256).cpu().numpy()
        else:
            heatmaps = predictions.cpu().numpy()
            coords = extract_coords_from_heatmap(heatmaps)
            
        ordered = order_points(coords)
        pts = np.array(ordered, np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img_resized, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in pts:
            cv2.circle(img_resized, tuple(pt[0]), radius=5, color=(255, 0, 0), thickness=-1)
            
        base_name = os.path.basename(image_path).split('.')[0]
        out_path = os.path.join(self.output_dir, f"{base_name}_{corner_version}_corners.jpg")
        
        final_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, final_bgr)
        return out_path

    def visualize_heatmaps(self, image_path, corner_version):
        import matplotlib.pyplot as plt
        print(f"Generating heatmap visuals for {corner_version}...")
        
        model_type, _ = self.get_corner_model_info(corner_version)
        if model_type == "direct":
            return None
            
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        corner_model, _ = self.load_corner_model(corner_version)
        
        corner_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        corner_tensor = corner_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps = corner_model(corner_tensor)[0].cpu().numpy()

        master_heatmap = np.max(heatmaps, axis=0)
        
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=100)
        axes = axes.flatten()
        
        axes[0].imshow(img_resized)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        for i in range(4):
            axes[i+1].imshow(heatmaps[i], cmap='magma')
            axes[i+1].set_title(f"Corner {i+1}")
            axes[i+1].axis('off')
            
        axes[5].imshow(master_heatmap, cmap='magma')
        axes[5].set_title("Master Heatmap")
        axes[5].axis('off')
        
        base_name = os.path.basename(image_path).split('.')[0]
        out_path = os.path.join(self.output_dir, f"{base_name}_{corner_version}_heatmaps.jpg")
        
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        plt.close()
        return out_path

    def process(self, image_path, corner_version=None, enhancer_version=None):
        base_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path)
        if img is None:
            print(f"Oops! Couldn't load {image_path}")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        current_img = img
        out_name = base_name

        if corner_version:
            print(f"Rectifying with {corner_version}...")
            corner_model, model_type = self.load_corner_model(corner_version)
            corners = self.detect_corners(img_rgb, corner_model, model_type)
            current_img = self.warp_image(img, corners)
            img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            out_name += f"_{corner_version}"

        if enhancer_version:
            print(f"Enhancing with {enhancer_version}...")
            enhancer_model = self.load_enhancer_model(enhancer_version)
            current_img = self.enhance_tiled(img_rgb, enhancer_model)
            out_name += f"_{enhancer_version}"

        if not corner_version and not enhancer_version:
            return None

        final_path = os.path.join(self.output_dir, f"{out_name}.jpg")
        cv2.imwrite(final_path, current_img)
        print(f"Success! Saved to {final_path}")
        return final_path