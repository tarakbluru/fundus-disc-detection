"""
Streamlit Web App for Interactive Disc Detection Parameter Tuning
Run with: streamlit run web_app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import glob
from PIL import Image
from src import image_processing

# Page configuration
st.set_page_config(
    page_title="Fundus Image Disc Detection Tuner",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Retinal Fundus Image - Optic Disc Detection Parameter Tuner")
st.markdown("Adjust parameters in real-time and see the effect on disc detection")

# Sidebar for parameters
st.sidebar.header("Detection Parameters")

# Get list of images
input_folder = "./input_images"
if not os.path.exists(input_folder):
    st.error(f"Input folder not found: {input_folder}")
    st.stop()

image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
    image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
    image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext.upper()}")))

if not image_files:
    st.error(f"No images found in {input_folder}")
    st.stop()

# Image selection
selected_image = st.sidebar.selectbox(
    "Select Image",
    options=[os.path.basename(f) for f in sorted(image_files)],
    index=0
)

# Parameter sliders
st.sidebar.subheader("Stage 1: Brightness Threshold")
disc_percentile = st.sidebar.slider(
    "Disc Brightness Percentile",
    min_value=70,
    max_value=99,
    value=90,
    step=1,
    help="Higher = only brightest pixels (more selective)"
)

st.sidebar.subheader("Stage 2: ROI Settings")
roi_expansion = st.sidebar.slider(
    "ROI Expansion (pixels)",
    min_value=100,
    max_value=400,
    value=200,
    step=25,
    help="Distance to expand from brightest point in each direction"
)

# Process button
if st.sidebar.button("🔄 Process Image", type="primary"):
    st.session_state.process_trigger = True

# Initialize session state
if 'process_trigger' not in st.session_state:
    st.session_state.process_trigger = False

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Original Image")
    # Load and display original image
    image_path = os.path.join(input_folder, selected_image)
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    st.image(original_rgb, use_container_width=True)

with col2:
    st.subheader("📊 Current Parameters")
    param_text = f"""
    - **Brightness Percentile:** {disc_percentile}
    - **ROI Expansion:** {roi_expansion}px (each direction)
    """
    st.markdown(param_text)

# Processing section
if st.session_state.process_trigger:
    with st.spinner("Processing image with current parameters..."):

        # Temporarily modify global parameters
        original_percentile = image_processing.DISC_BRIGHTNESS_PERCENTILE

        # Set new parameters
        image_processing.DISC_BRIGHTNESS_PERCENTILE = disc_percentile

        # Enable debug mode
        image_processing.DEBUG_MODE = True
        image_processing.DEBUG_STAGES['preprocessing'] = True
        image_processing.DEBUG_STAGES['disc_detection'] = True

        # Clear previous debug images
        debug_folder = "./debug_images"
        if os.path.exists(debug_folder):
            for f in glob.glob(os.path.join(debug_folder, "stage*.jpg")):
                try:
                    os.remove(f)
                except:
                    pass

        # Enable preprocessing debug
        image_processing.DEBUG_STAGES['preprocessing'] = True

        # Preprocess (automatically crops borders)
        enhanced, original_cropped = image_processing.preprocess_image(original_rgb)

        # Set ROI_EXPANSION parameter
        image_processing.ROI_EXPANSION = roi_expansion

        # ROI detection (returns None values - we stop at ROI threshold)
        _, _, _ = image_processing.detect_optic_disc(enhanced, original_cropped)

        # Restore original parameters
        image_processing.DISC_BRIGHTNESS_PERCENTILE = original_percentile

    # Display results
    st.divider()

    # Stage 1: Preprocessing
    st.subheader("📐 Stage 1: Preprocessing & Border Removal")

    # Row 0: Crop All Borders (in column layout)
    st.markdown("**Crop All Borders**")
    st1_row0 = st.columns(3)
    filepath = os.path.join(debug_folder, "stage1a_crop_all_borders.jpg")
    with st1_row0[0]:
        if os.path.exists(filepath):
            img = Image.open(filepath)
            st.image(img, use_container_width=True)
        else:
            st.warning("Not generated")

    st.markdown("---")

    # Row 1: Raw Channels (Blue, Green, Red)
    st.markdown("**Raw Channels (from cropped fundus)**")
    st1_row1 = st.columns(3)
    raw_channels = [
        ("stage1b_blue_channel.jpg", "Blue"),
        ("stage1c_green_channel.jpg", "Green"),
        ("stage1d_red_channel.jpg", "Red")
    ]
    for idx, (filename, title) in enumerate(raw_channels):
        filepath = os.path.join(debug_folder, filename)
        with st1_row1[idx]:
            st.markdown(f"**{title}**")
            if os.path.exists(filepath):
                img = Image.open(filepath)
                st.image(img, use_container_width=True)
            else:
                st.warning("Not generated")

    # Row 2: CLAHE Channels (Blue, Green, Red)
    st.markdown("**CLAHE Enhanced Channels**")
    st1_row2 = st.columns(3)
    clahe_channels = [
        ("stage1e_blue_clahe.jpg", "Blue CLAHE"),
        ("stage1f_green_clahe.jpg", "Green CLAHE"),
        ("stage1g_red_clahe.jpg", "Red CLAHE")
    ]
    for idx, (filename, title) in enumerate(clahe_channels):
        filepath = os.path.join(debug_folder, filename)
        with st1_row2[idx]:
            st.markdown(f"**{title}**")
            if os.path.exists(filepath):
                img = Image.open(filepath)
                st.image(img, use_container_width=True)
            else:
                st.warning("Not generated")

    # Row 3: Final Preprocessed (in column layout)
    st.markdown("**Final Preprocessed**")
    st1_row3 = st.columns(3)
    filepath = os.path.join(debug_folder, "stage1h_final_preprocessed.jpg")
    with st1_row3[0]:
        if os.path.exists(filepath):
            img = Image.open(filepath)
            st.image(img, use_container_width=True, caption=f"{img.size[0]}x{img.size[1]}px")
        else:
            st.warning("Not generated")

    st.divider()

    # Stage 2: ROI Detection
    st.subheader("🔍 Stage 2: ROI Detection (on cropped fundus)")
    stage2_files = [
        ("stage2a_brightest_point.jpg", "Brightest Point"),
        ("stage2b_roi_location.jpg", "ROI Location"),
        ("stage2c_roi_extracted.jpg", "ROI Extracted")
    ]

    # Display in 3 columns (3 images)
    st2_row1 = st.columns(3)

    for idx, (filename, title) in enumerate(stage2_files):
        filepath = os.path.join(debug_folder, filename)
        col = st2_row1[idx]

        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(filepath):
                img = Image.open(filepath)
                # For ROI extracted image, show at true size (no scaling)
                if filename == "stage2c_roi_extracted.jpg":
                    st.image(img, width=img.size[0], caption=f"Actual size: {img.size[0]}x{img.size[1]}px")
                else:
                    st.image(img, use_container_width=True)
            else:
                st.warning("Not generated")

    st.session_state.process_trigger = False

# Instructions
st.sidebar.divider()
st.sidebar.info("""
**How to use:**
1. Select an image from the dropdown
2. Adjust parameters using sliders:
   - **Brightness Percentile:** Threshold for bright regions (70-99)
   - **ROI Expansion:** Distance from brightest point (100-400px)
3. Click 'Process Image' to see results
4. View all stages:
   - **Stage 1:** Preprocessing (8 images) - crop all borders, channels, CLAHE, final preprocessed
   - **Stage 2:** ROI Detection (3 images) - find brightest point and extract ROI
5. Check where detection fails and adjust
6. Iterate until you find good parameters

**How it works:**
1. **Preprocessing pipeline (Stage 1)**
   - Crops ALL borders at once (black borders + white margins + dark borders)
   - Extracts all 3 channels (Blue, Green, Red) from clean fundus
   - Applies CLAHE to each channel independently
   - Uses GREEN channel CLAHE for ROI detection
   - Applies Gaussian blur for final preprocessing
2. **ROI detection (Stage 2) - works on clean fundus**
   - Finds brightest pixel in fundus (likely disc location)
   - Expands ROI from that point in all directions
   - Extracts ROI region
""")
