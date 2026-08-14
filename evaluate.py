import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ssim 
from torch.utils.data import DataLoader
from dataset import DocumentDataset, EnhancementDataset
from engine import DocumentScannerEngine, order_points, extract_coords_from_heatmap

engine = DocumentScannerEngine()

def save_table_as_png(headers, rows, title, filename):
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].get_text().set_color('white')
        table[(0, i)].get_text().set_weight('bold')
        
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    output_path = os.path.join(engine.output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visual table as {output_path}")

def append_to_markdown_report(title, headers, rows):
    md_path = os.path.join(engine.output_dir, "evaluation_results.md")
    
    with open(md_path, "a") as f:
        f.write(f"### {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")
        
    print(f"Appended results to {md_path}")

def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def evaluate_split(dataloader, model, device):
    total_baseline_psnr = 0.0
    total_baseline_ssim = 0.0
    total_model_psnr = 0.0
    total_model_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            baseline_psnr = calculate_psnr(inputs, targets)
            baseline_ssim = ssim(inputs, targets, data_range=1.0, size_average=True)
            
            total_baseline_psnr += baseline_psnr.item()
            total_baseline_ssim += baseline_ssim.item()
            
            if model is not None:
                preds = model(inputs)
                model_psnr = calculate_psnr(preds, targets)
                model_ssim = ssim(preds, targets, data_range=1.0, size_average=True)
                
                total_model_psnr += model_psnr.item()
                total_model_ssim += model_ssim.item()
                
            num_batches += 1
            
    avg_base_psnr = total_baseline_psnr / num_batches
    avg_base_ssim = total_baseline_ssim / num_batches
    avg_model_psnr = total_model_psnr / num_batches if model else 0.0
    avg_model_ssim = total_model_ssim / num_batches if model else 0.0
    
    return avg_base_psnr, avg_base_ssim, avg_model_psnr, avg_model_ssim

def generate_enhancer_table(enhancer_version="enhancer_v2"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Enhancer Evaluation on {device}...")
    
    try:
        model = engine.load_enhancer_model(enhancer_version)
        print(f"Loaded {enhancer_version} weights successfully!")
    except FileNotFoundError:
        print(f"Could not find weights for {enhancer_version}! Evaluating Baseline only.")
        model = None

    batch_size = 8
    base_dir = '/content/ready_dataset_enhancement'
    
    splits = {
        "Training": DataLoader(EnhancementDataset(f'{base_dir}/train'), batch_size=batch_size, shuffle=False),
        "Validation": DataLoader(EnhancementDataset(f'{base_dir}/val'), batch_size=batch_size, shuffle=False),
        "Test": DataLoader(EnhancementDataset(f'{base_dir}/test'), batch_size=batch_size, shuffle=False)
    }

    results = {}
    for split_name, loader in splits.items():
        print(f"Evaluating {split_name} split...")
        results[split_name] = evaluate_split(loader, model, device)

    headers = ["Split", "Baseline PSNR", "Model PSNR", "Baseline SSIM", "Model SSIM"]
    table_rows = []
    
    for split in ["Training", "Validation", "Test"]:
        b_psnr, b_ssim, m_psnr, m_ssim = results[split]
        row = [
            split,
            f"{b_psnr:.2f} dB",
            f"{m_psnr:.2f} dB",
            f"{b_ssim:.4f}",
            f"{m_ssim:.4f}"
        ]
        table_rows.append(row)

    print("\n" + "="*60)
    print("ENHANCEMENT MODEL PERFORMANCE TABLE")
    print("="*60)
    print(f"| {headers[0]:<12} | {headers[1]:<13} | {headers[2]:<10} | {headers[3]:<13} | {headers[4]:<10} |")
    print("-" * 72)
    for row in table_rows:
        print(f"| {row[0]:<12} | {row[1]:>13} | {row[2]:>10} | {row[3]:>13} | {row[4]:>10} |")
    print("="*60 + "\n")

    save_table_as_png(headers, table_rows, f"Enhancement Network Performance ({enhancer_version})", "enhancement_table.png")
    append_to_markdown_report(f"Enhancement Performance ({enhancer_version})", headers, table_rows)

def calculate_corner_errors(pred_coords, target_coords):
    pred_coords = np.array(pred_coords).reshape(4, 2)
    target_coords = target_coords.reshape(4, 2)
    
    pred_ordered = order_points(pred_coords)
    target_ordered = order_points(target_coords)
    
    distances = np.linalg.norm(pred_ordered - target_ordered, axis=1)
    
    mean_error = np.mean(distances)
    max_error = np.max(distances) 
    
    return mean_error, max_error

def evaluate_corner_model(model, model_type, dataloader, device, threshold=10.0):
    total_mean_error = 0.0
    successful_images = 0
    total_images = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            
            for i in range(images.size(0)):
                if model_type == "direct":
                    pred_coords = (predictions[i].cpu().numpy() * 256.0)
                else:
                    pred_coords = extract_coords_from_heatmap(predictions[i].cpu().numpy())
                    
                target_coords = (targets[i].cpu().numpy() * 256.0)
                mean_err, max_err = calculate_corner_errors(pred_coords, target_coords)
                
                total_mean_error += mean_err
                if max_err <= threshold:
                    successful_images += 1
                total_images += 1
                
    avg_error = total_mean_error / total_images
    success_rate = (successful_images / total_images) * 100.0
    
    return avg_error, success_rate

def generate_corner_table(direct_version="direct_v2", heatmap_version="heatmap_v2"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting Corner Model Showdown on {device}...")
    
    batch_size = 16
    test_loader = DataLoader(DocumentDataset('/content/ready_dataset/test'), batch_size=batch_size, shuffle=False)
    
    try:
        direct_model, _ = engine.load_corner_model(direct_version)
        direct_err, direct_succ = evaluate_corner_model(direct_model, "direct", test_loader, device)
    except FileNotFoundError:
        print(f"Could not find {direct_version}! Skipping.")
        direct_err, direct_succ = 0.0, 0.0

    try:
        heatmap_model, _ = engine.load_corner_model(heatmap_version)
        heatmap_err, heatmap_succ = evaluate_corner_model(heatmap_model, "heatmap", test_loader, device)
    except FileNotFoundError:
        print(f"Could not find {heatmap_version}! Skipping.")
        heatmap_err, heatmap_succ = 0.0, 0.0

    headers = ["Model Type", "Mean Pixel Error", "Success Rate (<10px)"]
    table_rows = [
        ["Direct Regression", f"{direct_err:.2f} px", f"{direct_succ:.1f} %"],
        ["Heatmap Regression", f"{heatmap_err:.2f} px", f"{heatmap_succ:.1f} %"]
    ]

    print("\n" + "="*60)
    print("CORNER DETECTION PERFORMANCE (Synthetic Test Set)")
    print("="*60)
    print(f"| {headers[0]:<20} | {headers[1]:<18} | {headers[2]:<20} |")
    print("-" * 64)
    for row in table_rows:
        print(f"| {row[0]:<20} | {row[1]:>18} | {row[2]:>20} |")
    print("="*60 + "\n")

    save_table_as_png(headers, table_rows, "Corner Detection Performance Comparison", "corner_table.png")
    append_to_markdown_report("Corner Detection Performance", headers, table_rows)

def evaluate_coco_corners(json_path, images_dir, corner_version):
    print(f"Evaluating corner error for {corner_version}...")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_map = {img['id']: img['file_name'] for img in coco_data['images']}

    engine = DocumentScannerEngine()
    model, model_type = engine.load_corner_model(corner_version)

    total_error = 0.0
    count = 0

    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in images_map:
            continue

        img_name = images_map[img_id]
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            continue

        seg = ann['segmentation'][0]
        gt_corners = np.array(seg).reshape(4, 2)
        gt_corners = order_points(gt_corners)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_corners = engine.detect_corners(img_rgb, model, model_type)

        error = np.linalg.norm(gt_corners - pred_corners, axis=1).mean()
        total_error += error
        count += 1

    if count == 0:
        print("No matching images found. Check your paths!")
        return

    avg_error = total_error / count
    print("=" * 40)
    print(f"Results for {corner_version}:")
    print(f"Images tested: {count}")
    print(f"Average Pixel Error: {avg_error:.2f} pixels")
    print("=" * 40)

if __name__ == "__main__":
    JSON_FILE = "/home/ho/Desktop/CV/scanit/data/Scanit.coco-segmentation/train/_annotations.coco.json"
    IMAGES_FOLDER = "/home/ho/Desktop/CV/scanit/data/Scanit.coco-segmentation/train/"
    
    evaluate_coco_corners(JSON_FILE, IMAGES_FOLDER, "direct_v3")
    evaluate_coco_corners(JSON_FILE, IMAGES_FOLDER, "heatmap_v3")