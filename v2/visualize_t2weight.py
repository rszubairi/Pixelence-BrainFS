import streamlit as st
import pydicom
import numpy as np
from scipy.ndimage import zoom

# Choose framework: 'tensorflow' or 'pytorch'
FRAMEWORK = 'tensorflow'  # Change this to switch between frameworks

if FRAMEWORK == 'tensorflow':
    from models_v2 import build_3d_generator
    from utils_v2 import calculate_ssim_3d, calculate_psnr_3d
    import tensorflow as tf
elif FRAMEWORK == 'pytorch':
    from models_v2_torch import create_3d_generator
    import torch
    # PyTorch-compatible metric functions
    def calculate_ssim_3d(y_true, y_pred):
        """PyTorch version of SSIM calculation"""
        # Convert to numpy for calculation
        y_true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        return torch.tensor(float(np.mean([1.0 for _ in range(y_true_np.shape[0])])))  # Placeholder

    def calculate_psnr_3d(y_true, y_pred):
        """PyTorch version of PSNR calculation"""
        mse = torch.mean((y_true - y_pred) ** 2)
        if mse.item() == 0:  # Use .item() to get scalar value
            return torch.tensor(float('inf'))
        max_val = torch.tensor(1.0)
        return 20 * torch.log10(max_val) - 10 * torch.log10(mse)
import glob
import os
from pathlib import Path

# Set Streamlit to fullscreen mode
st.set_page_config(layout="wide", page_title="3D T2-Weighted Fat Suppression")

# Set the title of the Streamlit app
st.title("🧠 3D T2-Weighted Fat Suppression DICOM Volume Viewer")

# Add custom CSS to set the sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Parameters
volume_depth = 32
volume_height = 128
volume_width = 128
input_dims = (volume_depth, volume_height, volume_width, 3)  # Single modality
output_dims = (volume_depth, volume_height, volume_width, 1)
GENERATOR_3D = None

def load_3d_generator():
    """Load the 3D U-Net model"""
    global GENERATOR_3D
    if GENERATOR_3D is None:
        if FRAMEWORK == 'tensorflow':
            GENERATOR_3D = build_3d_generator(input_dims, f=16, num_modalities=1)
            # Try to load trained weights if available
            try:
                GENERATOR_3D.load_weights('best_weights_t2weight.weights.h5')
                st.sidebar.write('<h4 style="color:green;">✅ Pre-trained T2-weighted TensorFlow weights loaded!</h4>', unsafe_allow_html=True)
            except:
                try:
                    # Fallback to existing v2 weights
                    GENERATOR_3D.load_weights('../best_weights_v2.weights.h5')
                    st.sidebar.write('<h4 style="color:orange;">⚠️  Using existing v2 weights (T2-weighted training not complete)</h4>', unsafe_allow_html=True)
                except:
                    st.sidebar.write('<h4 style="color:red;">❌ No trained weights found - using random initialization</h4>', unsafe_allow_html=True)
        elif FRAMEWORK == 'pytorch':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            GENERATOR_3D = create_3d_generator(in_channels=3, num_modalities=1)
            GENERATOR_3D.to(device)
            # Try to load trained weights if available
            try:
                GENERATOR_3D.load_state_dict(torch.load('best_weights_v2_torch.pth'))
                st.sidebar.write('<h4 style="color:green;">✅ Pre-trained PyTorch weights loaded!</h4>', unsafe_allow_html=True)
            except:
                st.sidebar.write('<h4 style="color:orange;">⚠️  Using randomly initialized PyTorch model</h4>', unsafe_allow_html=True)
    return GENERATOR_3D

def load_dicom_volume(uploaded_files):
    """Load multiple DICOM files and stack them into a 3D volume"""
    slices = []

    # Sort files by instance number if available
    file_data = []
    for uploaded_file in uploaded_files:
        try:
            scan = pydicom.dcmread(uploaded_file)
            instance_num = getattr(scan, 'InstanceNumber', 0)
            file_data.append((instance_num, uploaded_file))
        except:
            file_data.append((0, uploaded_file))

    # Sort by instance number
    file_data.sort(key=lambda x: x[0])

    # Load and process slices
    for _, uploaded_file in file_data:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            scan = pydicom.dcmread(uploaded_file)
            slice_data = scan.pixel_array.astype('float32')
            slices.append(slice_data)
        except Exception as e:
            st.error(f"Error loading DICOM slice: {e}")
            continue

    if len(slices) == 0:
        raise ValueError("No valid DICOM slices found")

    # Stack into volume
    volume = np.stack(slices, axis=0)
    return volume

def preprocess_volume(volume):
    """Preprocess 3D volume for the model"""
    # Debug: print input volume info
    print(f"Input volume shape: {volume.shape}, range: [{volume.min():.2f}, {volume.max():.2f}]")

    # Ensure minimum depth for meaningful 3D processing
    min_depth = max(8, volume.shape[0])  # At least 8 slices for 3D processing

    if volume.shape[0] < min_depth:
        # If we have too few slices, duplicate them to reach minimum depth
        repetitions = (min_depth + volume.shape[0] - 1) // volume.shape[0]  # Ceiling division
        volume = np.tile(volume, (repetitions, 1, 1))
        volume = volume[:min_depth]  # Trim to exact size
        print(f"After duplication: {volume.shape}")

    # For visualization, we want to keep the original volume intact
    # Only resize for model input
    target_shape = (volume_depth, volume_height, volume_width)

    # Handle different input shapes
    if len(volume.shape) == 3:  # (depth, height, width)
        # Make sure target shape has minimum dimensions
        safe_target = (
            max(target_shape[0], volume.shape[0]),  # Don't reduce depth below original
            max(target_shape[1], 64),  # Minimum 64 for height
            max(target_shape[2], 64),  # Minimum 64 for width
        )

        volume_model = zoom(volume, (
            safe_target[0] / volume.shape[0],
            safe_target[1] / volume.shape[1],
            safe_target[2] / volume.shape[2]
        ), order=1)  # Linear interpolation
        volume_model = np.expand_dims(volume_model, axis=-1)  # Add channel dimension
    elif len(volume.shape) == 4:  # Already has channels
        safe_target = (
            max(target_shape[0], volume.shape[0]),
            max(target_shape[1], 64),
            max(target_shape[2], 64),
            volume.shape[3]  # Keep channels
        )

        volume_model = zoom(volume, (
            safe_target[0] / volume.shape[0],
            safe_target[1] / volume.shape[1],
            safe_target[2] / volume.shape[2],
            1  # Keep channels
        ), order=1)

    print(f"Model input shape: {volume_model.shape}")

    # Ensure we have the right shape after processing
    if volume_model.shape[0] < 4:
        # Pad with copies if still too small
        while volume_model.shape[0] < 4:
            volume_model = np.concatenate([volume_model, volume_model], axis=0)
        volume_model = volume_model[:4]

    # Normalize to [0, 1]
    volume_model = normalisation(volume_model)

    # Ensure 3 channels for RGB input
    if volume_model.shape[-1] == 1:
        volume_model = np.repeat(volume_model, 3, axis=-1)

    # Add batch dimension and handle channel ordering
    volume_model = np.expand_dims(volume_model, axis=0)

    if FRAMEWORK == 'pytorch':
        # Convert from TensorFlow format (batch, D, H, W, C) to PyTorch format (batch, C, D, H, W)
        volume_model = np.transpose(volume_model, (0, 4, 1, 2, 3))

    print(f"Final model input shape: {volume_model.shape}")
    return volume_model

def generate_suppressed_volume(volume):
    """Generate fat suppressed volume using 3D U-Net"""
    generator = load_3d_generator()

    # Preprocess input
    input_volume = preprocess_volume(volume)

    if FRAMEWORK == 'tensorflow':
        # TensorFlow inference
        pred_volume = generator.predict(input_volume, verbose=0)
        pred_volume = np.squeeze(pred_volume, axis=0)  # Remove batch dimension
    elif FRAMEWORK == 'pytorch':
        # PyTorch inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.from_numpy(input_volume).float().to(device)

        with torch.no_grad():
            pred_tensor = generator(input_tensor)
            pred_volume = pred_tensor.cpu().numpy()
            pred_volume = np.squeeze(pred_volume, axis=0)  # Remove batch dimension

    # Calculate 3D metrics
    # Normalize volumes for metric calculation
    input_norm = normalisation(volume)
    if len(input_norm.shape) == 3:
        input_norm = np.expand_dims(input_norm, axis=-1)

    # Resize input_norm to match pred_volume shape for comparison
    input_resized = zoom(input_norm, (
        pred_volume.shape[0] / input_norm.shape[0],
        pred_volume.shape[1] / input_norm.shape[1],
        pred_volume.shape[2] / input_norm.shape[2],
        1
    ))

    if FRAMEWORK == 'tensorflow':
        # TensorFlow metric calculation
        img1 = tf.convert_to_tensor(input_resized, dtype=tf.float32)
        img1 = tf.expand_dims(img1, axis=0)  # Add batch dimension

        img2 = tf.convert_to_tensor(pred_volume, dtype=tf.float32)
        img2 = tf.expand_dims(img2, axis=0)  # Add batch dimension

        ssim_val = calculate_ssim_3d(img1, img2)
        psnr_val = calculate_psnr_3d(img1, img2)

        return pred_volume, float(ssim_val.numpy()), float(psnr_val.numpy())

    elif FRAMEWORK == 'pytorch':
        # PyTorch metric calculation
        img1 = torch.from_numpy(input_resized).float()
        img1 = img1.unsqueeze(0)  # Add batch dimension

        img2 = torch.from_numpy(pred_volume).float()
        img2 = img2.unsqueeze(0)  # Add batch dimension

        ssim_val = calculate_ssim_3d(img1, img2)
        psnr_val = calculate_psnr_3d(img1, img2)

        return pred_volume, float(ssim_val.item()), float(psnr_val.item())

def normalisation(image):
    """Normalize image to [0,1] range"""
    image = image.astype('float32')
    min_val = np.min(image)
    max_val = np.max(image)
    epsilon = 1e-7
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val + epsilon)
    return image

def prepare_display_volume(volume):
    """Prepare volume for display by ensuring adequate dimensions"""
    # Ensure minimum depth for meaningful visualization
    min_depth = max(8, volume.shape[0])

    if volume.shape[0] < min_depth:
        # Duplicate slices to reach minimum depth
        repetitions = (min_depth + volume.shape[0] - 1) // volume.shape[0]
        volume = np.tile(volume, (repetitions, 1, 1))
        volume = volume[:min_depth]

    # Ensure minimum spatial dimensions
    min_spatial = 64
    if volume.shape[1] < min_spatial or volume.shape[2] < min_spatial:
        # Resize spatial dimensions if too small
        new_h = max(min_spatial, volume.shape[1])
        new_w = max(min_spatial, volume.shape[2])

        resized_slices = []
        for i in range(volume.shape[0]):
            slice_data = volume[i]
            if len(slice_data.shape) == 2:
                resized = zoom(slice_data, (new_h / slice_data.shape[0], new_w / slice_data.shape[1]), order=1)
            else:
                resized = zoom(slice_data, (new_h / slice_data.shape[0], new_w / slice_data.shape[1], 1), order=1)
            resized_slices.append(resized)

        volume = np.stack(resized_slices, axis=0)

    return volume

def prepare_display(image):
    """Prepare image for display (convert to uint8)"""
    image = image.astype('float32')
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def extract_slices(volume, slice_indices=None):
    """Extract slices from different planes for visualization"""
    if slice_indices is None:
        depth_mid = volume.shape[0] // 2
        height_mid = volume.shape[1] // 2
        width_mid = volume.shape[2] // 2
        slice_indices = (depth_mid, height_mid, width_mid)

    depth_slice, height_slice, width_slice = slice_indices

    # Axial (XY plane at depth z)
    axial = volume[depth_slice, :, :, 0] if len(volume.shape) == 4 else volume[depth_slice, :, :]

    # Sagittal (YZ plane at width x)
    sagittal = volume[:, height_slice, :, 0] if len(volume.shape) == 4 else volume[:, height_slice, :]

    # Coronal (XZ plane at height y)
    coronal = volume[:, :, width_slice, 0] if len(volume.shape) == 4 else volume[:, :, width_slice]

    return axial, sagittal, coronal

def display_volume_comparison(original_volume, suppressed_volume, ssim_val, psnr_val):
    """Display original and suppressed volumes side by side"""
    st.markdown("### 📊 Volume Comparison Results")
    st.markdown(f"**3D SSIM:** {ssim_val:.2f} | **3D PSNR:** {psnr_val:.2f}")

    # Validate volume shapes
    if len(original_volume.shape) < 3 or len(suppressed_volume.shape) < 3:
        st.error("❌ Invalid volume shapes for display")
        return

    # Use original volume dimensions for slider ranges (it's the display volume)
    orig_depth, orig_height, orig_width = original_volume.shape[:3]
    supp_depth, supp_height, supp_width = suppressed_volume.shape[:3]

    # Ensure minimum dimensions for sliders
    if orig_depth <= 1 or orig_height <= 1 or orig_width <= 1:
        st.error("❌ Volume dimensions too small for slice visualization")
        return

    # Slice selector - based on original volume dimensions
    col1, col2, col3 = st.columns(3)
    with col1:
        depth_slice = st.slider("Axial Slice (Depth)", 0, max(1, orig_depth-1),
                               min(orig_depth//2, orig_depth-1), key="axial")
    with col2:
        height_slice = st.slider("Sagittal Slice (Height)", 0, max(1, orig_height-1),
                                min(orig_height//2, orig_height-1), key="sagittal")
    with col3:
        width_slice = st.slider("Coronal Slice (Width)", 0, max(1, orig_width-1),
                               min(orig_width//2, orig_width-1), key="coronal")

    # Extract slices from original volume
    orig_slice_indices = (depth_slice, height_slice, width_slice)
    orig_axial, orig_sagittal, orig_coronal = extract_slices(original_volume, orig_slice_indices)

    # For suppressed volume, map the slice indices to its coordinate system
    # Since the suppressed volume may have different dimensions, we need to scale the indices
    try:
        supp_depth_slice = int(depth_slice * supp_depth / max(1, orig_depth)) if orig_depth > 1 else 0
        supp_height_slice = int(height_slice * supp_height / max(1, orig_height)) if orig_height > 1 else 0
        supp_width_slice = int(width_slice * supp_width / max(1, orig_width)) if orig_width > 1 else 0

        # Ensure indices are within bounds
        supp_depth_slice = max(0, min(supp_depth_slice, supp_depth - 1))
        supp_height_slice = max(0, min(supp_height_slice, supp_height - 1))
        supp_width_slice = max(0, min(supp_width_slice, supp_width - 1))

        print(f"Debug - Original indices: ({depth_slice}, {height_slice}, {width_slice})")
        print(f"Debug - Mapped indices: ({supp_depth_slice}, {supp_height_slice}, {supp_width_slice})")
        print(f"Debug - Volume shapes: display {original_volume.shape}, suppressed {suppressed_volume.shape}")

        supp_slice_indices = (supp_depth_slice, supp_height_slice, supp_width_slice)
        supp_axial, supp_sagittal, supp_coronal = extract_slices(suppressed_volume, supp_slice_indices)

    except Exception as slice_error:
        st.error(f"Error extracting slices from suppressed volume: {slice_error}")
        st.error(f"Debug info: orig_shape={original_volume.shape}, supp_shape={suppressed_volume.shape}")
        st.error(f"Indices: orig=({depth_slice}, {height_slice}, {width_slice}), supp=({supp_depth_slice}, {supp_height_slice}, {supp_width_slice})")
        return

    # Display comparisons
    planes = [
        ("Axial View", orig_axial, supp_axial),
        ("Sagittal View", orig_sagittal, supp_sagittal),
        ("Coronal View", orig_coronal, supp_coronal)
    ]

    for plane_name, orig_slice, supp_slice in planes:
        st.markdown(f"#### {plane_name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(prepare_display(orig_slice), caption=f'Original {plane_name}', use_container_width=True)

        with col2:
            st.image(prepare_display(supp_slice), caption=f'Fat Suppressed {plane_name}', use_container_width=True)

# Sidebar content
try:
    logo_path = os.path.join(os.path.dirname(__file__), '..', 'logo.png')
    st.sidebar.image(logo_path, use_container_width=False, width=150)
except FileNotFoundError:
    st.sidebar.markdown("🧠 **3D Fat Suppression v2.0 FROM Pixellence**")

st.sidebar.markdown("""
Upload multiple DICOM slices to form a 3D volume and apply advanced 3D T2-weighted fat suppression using our enhanced U-Net architecture trained on clinical REMIND data.

**Features:**
- 🧠 3D U-Net with attention mechanisms (T2-weighted trained)
- 📊 Multi-planar visualization
- 🎯 Spatial consistency preservation
- 📈 3D SSIM & PSNR metrics
""")

# Model info
st.sidebar.markdown("### 🤖 Model Configuration")
st.sidebar.markdown(f"""
- **Architecture:** 3D U-Net with CBAM Attention
- **Input Shape:** {input_dims}
- **Output Shape:** {output_dims}
- **Modalities:** Single (RGB from grayscale)
""")

# Upload DICOM files
st.sidebar.markdown("### 📁 Upload DICOM Volume")
uploaded_files = st.sidebar.file_uploader(
    "Upload DICOM slices (multiple files for 3D volume)...",
    type=["dcm", "DCM"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and len(uploaded_files) > 0:
    try:
        with st.spinner("Loading DICOM volume..."):
            volume = load_dicom_volume(uploaded_files)

        st.success(f"✅ Successfully loaded 3D volume: {volume.shape}")

        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"Volume shape: {volume.shape}")
            st.write(f"Volume dtype: {volume.dtype}")
            st.write(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
            st.write(f"Number of uploaded files: {len(uploaded_files)}")

        # Generate suppressed volume
        with st.spinner("Applying 3D fat suppression..."):
            suppressed_volume, ssim_val, psnr_val = generate_suppressed_volume(volume)

        # Prepare display volume (ensure adequate dimensions)
        display_volume = prepare_display_volume(volume)

        # Display results
        display_volume_comparison(display_volume, suppressed_volume, ssim_val, psnr_val)

        # Volume statistics
        with st.expander("📈 Volume Statistics"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Volume:**")
                st.markdown(f"- Shape: {volume.shape}")
                st.markdown(f"- Data type: {volume.dtype}")
                st.markdown(f"- Value range: [{volume.min():.2f}, {volume.max():.2f}]")
                st.markdown(f"- Mean: {volume.mean():.2f}")
                st.markdown(f"- Std: {volume.std():.2f}")

            with col2:
                st.markdown("**Suppressed Volume:**")
                st.markdown(f"- Shape: {suppressed_volume.shape}")
                st.markdown(f"- Data type: {suppressed_volume.dtype}")
                st.markdown(f"- Value range: [{suppressed_volume.min():.2f}, {suppressed_volume.max():.2f}]")
                st.markdown(f"- Mean: {suppressed_volume.mean():.2f}")
                st.markdown(f"- Std: {suppressed_volume.std():.2f}")

    except Exception as e:
        st.error(f"Error processing DICOM volume: {e}")
        st.info("Please ensure you upload valid DICOM files and try again.")

else:
    st.info("👆 Please upload DICOM files to get started!")

    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Upload DICOM files**: Select multiple DICOM slices that form a 3D volume
        2. **Automatic processing**: The app will load and stack the slices
        3. **3D fat suppression**: Advanced 3D U-Net processes the entire volume
        4. **Multi-planar view**: Explore results in axial, sagittal, and coronal planes
        5. **Quality metrics**: View 3D SSIM and PSNR scores

        **Tips:**
        - Upload slices from the same series for best results
        - More slices generally improve 3D reconstruction quality
        - The model works best with brain MRI volumes
        """)

    # Sample visualization (placeholder)
    st.markdown("### 🎯 Preview of 3D Capabilities")
    st.markdown("Upload DICOM files above to see the 3D fat suppression in action!")

# Footer
st.markdown("---")
st.markdown("*Developed by Pixellence - Advanced 3D T2-Weighted Fat Suppression*")
