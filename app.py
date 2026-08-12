import os
import cv2
import streamlit as st
from PIL import Image, ImageOps
from engine import DocumentScannerEngine

@st.cache_resource
def get_engine():
    return DocumentScannerEngine()

engine = get_engine()

def get_available_models():
    """Dynamically scans the models directory for BOTH heatmap and direct checkpoints!"""
    models_dir = "models"
    corner_models, enhancer_models = [], []
    
    if os.path.exists(models_dir):
        for folder in sorted(os.listdir(models_dir)):
            folder_path = os.path.join(models_dir, folder)
            if os.path.isdir(folder_path):
                
                if os.path.exists(os.path.join(folder_path, "heatmap_best.pth")) or \
                   os.path.exists(os.path.join(folder_path, "direct_best.pth")):
                    corner_models.append(folder)
                if os.path.exists(os.path.join(folder_path, "enhancer_best.pth")):
                    enhancer_models.append(folder)
                    
    return ["None"] + corner_models, ["None"] + enhancer_models

st.set_page_config(page_title="Scanit", layout="wide")

st.title("Scanit: Deep Learning Document Scanner")
st.markdown("Upload a raw document photograph to process it through the enhancement pipeline.")


st.sidebar.header("Model Settings")

available_corners, available_enhancers = get_available_models()

corner_model = st.sidebar.selectbox(
    "Corner Detection Model",
    available_corners,
    help="Select the model used to find the 4 corners of the document."
)

enhancer_model = st.sidebar.selectbox(
    "Enhancement Model",
    available_enhancers,
    help="Select the U-Net model used to flatten lighting and sharpen text."
)


uploaded_file = st.file_uploader("Upload document photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_upload_path = "temp_uploaded_image.jpg"
    
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image.convert("RGB").save(temp_upload_path)
        
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.image(temp_upload_path, caption="Raw Input Photo", use_container_width=True)
    
    
    if st.button("Process Document", use_container_width=True):
        corner_opt = None if corner_model == "None" else corner_model
        enhancer_opt = None if enhancer_model == "None" else enhancer_model
        
        with st.spinner("Processing document..."):
            final_output_path = engine.process(temp_upload_path, corner_version=corner_opt, enhancer_version=enhancer_opt)
            
            heatmap_path = None
            corners_path = None
            
            
            if corner_opt:
                corners_path = engine.visualize_corners(temp_upload_path, corner_opt)
                heatmap_path = engine.visualize_heatmaps(temp_upload_path, corner_opt)

        st.success("Processing Complete")
        
        
        tab_names = ["Final Scan", "Before & After"]
        if corners_path:
            tab_names.append("Corner Detection")
        if heatmap_path:  
            tab_names.append("Model Heatmaps")
            
        active_tabs = st.tabs(tab_names)
        
        
        with active_tabs[0]:
            if final_output_path:
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(final_output_path, caption="Restored Document", use_container_width=True)
            else:
                st.info("No models selected to generate a final scan.")
                
        
        with active_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.image(temp_upload_path, caption="Raw Photo", use_container_width=True)
            with col2:
                if final_output_path:
                    st.image(final_output_path, caption="Enhanced Scan", use_container_width=True)
                else:
                    st.info("Select an Enhancement model to see the output.")
            
        
        tab_idx = 2
        if corners_path:
            with active_tabs[tab_idx]:
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(corners_path, caption="Predicted Corner Keypoints", use_container_width=True)
            tab_idx += 1

        
        if heatmap_path:
            with active_tabs[tab_idx]:
                c1, c2, c3 = st.columns([1, 4, 1])
                with c2:
                    st.image(heatmap_path, caption="Network Heatmap Activations", use_container_width=True)

        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)