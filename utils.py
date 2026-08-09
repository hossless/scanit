import cv2
import numpy as np
import random


def get_random_perspective_corners(bg_w, bg_h):
    
    margin = 100 
    wiggle_room = 60  
    
    base_corners = [
        [margin, margin],                         
        [bg_w - margin, margin],                  
        [bg_w - margin, bg_h - margin],           
        [margin, bg_h - margin]                   
    ]
    
    randomized_corners = []
    for x, y in base_corners:
        rand_x = x + random.randint(-wiggle_room, wiggle_room)
        rand_y = y + random.randint(-wiggle_room, wiggle_room)
        randomized_corners.append([rand_x, rand_y])
        
    return np.float32(randomized_corners)

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


def generate_synthetic_sample(clean_scan, background_img):    
    warped_composite, corners = warp_scan_to_background(clean_scan, background_img)
    
    
    img = apply_resolution_loss(warped_composite)
    
    img_float = img.astype(np.float32)
    img_float = apply_color_and_lighting(img_float)
    img_float = apply_shadows_and_gradients(img_float)
    
    img_uint8 = apply_blur_and_noise(img_float)
    final_img = apply_jpeg_compression(img_uint8)
    
    return final_img, corners