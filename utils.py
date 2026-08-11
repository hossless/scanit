import cv2
import torch
import random
import numpy as np

def get_random_perspective_corners(bg_w, bg_h):
    scale = random.uniform(0.3, 0.9)
    doc_w = bg_w * scale
    doc_h = bg_h * scale

    cx = random.randint(int(doc_w/2), int(bg_w - doc_w/2))
    cy = random.randint(int(doc_h/2), int(bg_h - doc_h/2))

    half_w, half_h = doc_w / 2, doc_h / 2
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    angle = random.uniform(-75, 75)
    theta = np.radians(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix.T)
    shifted_corners = rotated_corners + np.array([cx, cy])

    wiggle = int(min(bg_w, bg_h) * 0.05) 
    for i in range(4):
        shifted_corners[i][0] += random.randint(-wiggle, wiggle)
        shifted_corners[i][1] += random.randint(-wiggle, wiggle)

    shifted_corners[:, 0] = np.clip(shifted_corners[:, 0], 0, bg_w)
    shifted_corners[:, 1] = np.clip(shifted_corners[:, 1], 0, bg_h)

    return np.float32(shifted_corners)

def warp_scan_to_background(clean_scan, background_img):
    scan_h, scan_w = clean_scan.shape[:2]
    bg_h, bg_w = background_img.shape[:2]
    
    src_points = np.float32([
        [0, 0], [scan_w, 0], [scan_w, scan_h], [0, scan_h]
    ])
    
    dst_points = get_random_perspective_corners(bg_w, bg_h)
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_scan = cv2.warpPerspective(clean_scan, matrix, (bg_w, bg_h))
    
    mask = np.ones((scan_h, scan_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, matrix, (bg_w, bg_h))
    
    background_copy = background_img.copy()
    background_copy[warped_mask == 255] = warped_scan[warped_mask == 255]
    
    return background_copy, dst_points

def apply_resolution_loss(img):
    h, w = img.shape[:2]
    scale_factor = random.uniform(2.0, 4.0)
    small_img = cv2.resize(img, (int(w / scale_factor), int(h / scale_factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)

def apply_color_and_lighting(img_float):
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-40, 40)
    img_float = cv2.convertScaleAbs(img_float, alpha=alpha, beta=beta).astype(np.float32)
    
    r_scale = random.uniform(0.8, 1.2)
    b_scale = random.uniform(0.8, 1.2)
    img_float[:, :, 2] *= r_scale  
    img_float[:, :, 0] *= b_scale 
    
    return np.clip(img_float, 0, 255)

def apply_shadows_and_gradients(img_float):
    h, w = img_float.shape[:2]
    shadow_mask = np.ones((h, w), dtype=np.float32)
    
    num_points = random.randint(3, 5)
    points = [[random.randint(0, w), random.randint(0, h)] for _ in range(num_points)]
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    
    shadow_intensity = random.uniform(0.3, 0.7) 
    cv2.fillPoly(shadow_mask, [pts], shadow_intensity)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (101, 101), 50)
    
    for c in range(3):
        img_float[:, :, c] *= shadow_mask
        
    return np.clip(img_float, 0, 255)

def apply_blur_and_noise(img_float):
    blur_radius = random.choice([3, 5])
    img_float = cv2.GaussianBlur(img_float, (blur_radius, blur_radius), 0)
    
    noise = np.random.normal(0, random.uniform(2, 10), img_float.shape)
    img_float = img_float + noise
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_jpeg_compression(img_uint8):
    quality = random.randint(30, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img_uint8, encode_param)
    return cv2.imdecode(encoded_img, 1)

def order_points(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices]

def generate_synthetic_sample(clean_scan, background_img):    
    warped_composite, corners = warp_scan_to_background(clean_scan, background_img)
    
    img = apply_resolution_loss(warped_composite)
    img_float = img.astype(np.float32)
    img_float = apply_color_and_lighting(img_float)
    img_float = apply_shadows_and_gradients(img_float)
    img_uint8 = apply_blur_and_noise(img_float)
    final_img = apply_jpeg_compression(img_uint8)
    
    ordered_corners = order_points(corners)
    
    return final_img, ordered_corners

def generate_target_heatmaps(corners_batch, image_size=256, sigma=7.0):
    batch_size = corners_batch.shape[0]
    heatmaps = torch.zeros((batch_size, 4, image_size, image_size), device=corners_batch.device)
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(image_size, device=corners_batch.device), 
        torch.arange(image_size, device=corners_batch.device), 
        indexing='ij'
    )
    
    for b in range(batch_size):
        coords = corners_batch[b].view(4, 2) * image_size 
        for i in range(4):
            x, y = coords[i][0], coords[i][1]
            dist_sq = (x_grid - x)**2 + (y_grid - y)**2
            heatmaps[b, i] = torch.exp(-dist_sq / (2 * sigma**2))
            
    return heatmaps

def generate_enhancement_sample(clean_scan, background_img):
    
    scan_h, scan_w = clean_scan.shape[:2]
    degraded_photo, corners = generate_synthetic_sample(clean_scan, background_img)
    
    
    
    src_points = corners
    dst_points = np.float32([
        [0, 0], 
        [scan_w, 0], 
        [scan_w, scan_h], 
        [0, scan_h]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    rectified_degraded = cv2.warpPerspective(degraded_photo, matrix, (scan_w, scan_h))
    
    
    final_input = cv2.resize(rectified_degraded, (512, 512))
    final_target = cv2.resize(clean_scan, (512, 512))
    
    return final_input, final_target