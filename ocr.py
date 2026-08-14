import os
import cv2
import pytesseract
from pytesseract import Output
from engine import DocumentScannerEngine

def evaluate_ocr_confidence(image_path, lang='fas+eng', rotate_left=False):
    if not os.path.exists(image_path):
        return 0.0, "[File Not Found]"
        
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, "[Cannot Read Image]"

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if rotate_left:
        img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    ocr_data = pytesseract.image_to_data(img_gray, lang=lang, output_type=Output.DICT)
    
    confidences = []
    extracted_words = []

    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        
        if conf != -1 and len(word) > 0:
            confidences.append(conf)
            extracted_words.append(word)

    if not confidences:
        return 0.0, "[No Text Detected]"

    avg_conf = sum(confidences) / len(confidences)
    full_text = " ".join(extracted_words)
    
    return avg_conf, full_text

if __name__ == "__main__":
    print("Booting up the Engine...")
    engine = DocumentScannerEngine()
    
    raw_path = "data/real/pic8.jpg"
    camscanner_path = "data/real/scan8.jpg"
    
    corner_model = "heatmap_v3"
    enhancer_model = "enhancer_v2"
    
    print("\nGenerating Version A (Warped Only)...")
    version_a_path = engine.process(raw_path, corner_version=corner_model, enhancer_version=None)
    
    print("Generating Version B (Warped + Enhanced)...")
    version_b_path = engine.process(raw_path, corner_version=corner_model, enhancer_version=enhancer_model, use_tiling=True)
    
    print("\nRunning Tesseract OCR Showdown...\n")
    
    conf_a, text_a = evaluate_ocr_confidence(version_a_path, rotate_left=True)
    conf_b, text_b = evaluate_ocr_confidence(version_b_path, rotate_left=True)
    conf_c, text_c = evaluate_ocr_confidence(camscanner_path, rotate_left=False)
    
    print("="*60)
    print("OCR CONFIDENCE SHOWDOWN (pic8)")
    print("="*60)
    print(f"Version A (Raw Warped):      {conf_a:>5.2f}%")
    print(f"Version B (Enhanced):        {conf_b:>5.2f}%")
    print(f"Version C (CamScanner):      {conf_c:>5.2f}%")
    print("="*60)
    