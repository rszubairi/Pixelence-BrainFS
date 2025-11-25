import streamlit as st
import pydicom
import numpy as np
from scipy.ndimage import zoom
from models import build_generator
import tensorflow as tf

# Set Streamlit to fullscreen mode
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("Fat Suppression DICOM Image Viewer")

# Add custom CSS to set the sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Parameters
input_dims = (256, 256, 3)
output_dims = (256, 256, 1)
GENERATOR = None

def generate_suppressed_image(image):
    global GENERATOR
    if GENERATOR is None:
        GENERATOR = build_generator(input_dims, 22, True)
    st.sidebar.write('<h3 style="color:red;">Fat Suppression Model Loaded Successfully</h3>', unsafe_allow_html=True)

    im_sup = normalisation(image)
    if im_sup.ndim == 2:
        im_sup = np.expand_dims(im_sup, axis=-1)  # Add channel dimension (256,256,1)
    im_sup = np.repeat(im_sup, 3, axis=-1)  # Repeat to 3 channels (256,256,3)
    im_sup = np.expand_dims(im_sup, 0)  # Add batch dimension (1,256,256,3)
    pred = GENERATOR.predict(im_sup, verbose=0)
    pred = np.squeeze(pred, 0)  # Remove batch dimension
    pred = convert_uint8(pred)  # Convert to uint8 for display

    # Calculate metrics between normalized original and predicted
    pred_norm = pred.astype('float32') / 255.0  # 0-1
    im_sup_norm = normalisation(image)  # original normalized 0-1
    img1 = tf.convert_to_tensor(im_sup_norm, dtype=tf.float32)
    img1 = tf.expand_dims(img1, axis=0)  # add batch
    img1 = tf.expand_dims(img1, axis=3)  # add channel
    img2 = tf.convert_to_tensor(pred_norm, dtype=tf.float32)
    img2 = tf.expand_dims(img2, axis=0)  # pred_norm already has channel dim
    ssim_val = calculate_ssim(img1, img2)
    psnr_val = calculate_psnr(img1, img2)

    return pred, float(ssim_val.numpy()), float(psnr_val.numpy())

def normalisation(image):
    image = image.astype('float32')
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the pixel values
    epsilon = 1e-7
    image = (image - min_val) / (max_val - min_val + epsilon)
    return image

def rescale(arr, dim=256):
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, -1)
    new_ar = zoom(arr, (dim/arr.shape[0], dim/arr.shape[1], 1))
    if arr.shape[-1] == 1:
        new_ar = new_ar.squeeze(-1)
    return new_ar

def convert_uint8(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return image

def prepare_display(image):
    image = image.astype('float32')
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def calculate_ssim(image1, image2):
    return tf.reduce_mean(tf.image.ssim(image1, image2, max_val=1.0, filter_size=5)) * 100

def calculate_psnr(image1, image2):
    return tf.reduce_mean(tf.image.psnr(image1, image2, max_val=1.0))

# Sidebar content

logo_path = "logo.png"  # Replace with your logo image file path or comment out if missing
try:
    st.sidebar.image(logo_path, use_container_width=False, width=150)
except FileNotFoundError:
    st.sidebar.markdown("🧠 **Fat Suppression V1.0.3 FROM Pixellence**")

st.sidebar.write("Upload a DICOM file to apply Fat Suppression and view the result.")

# Upload DICOM
uploaded_file = st.file_uploader("Please Upload the DICOM scan...", type=["dcm"])

# Check if a file is uploaded
if uploaded_file is not None:
    scan = pydicom.dcmread(uploaded_file)
    scan_im = scan.pixel_array
    scan_im = rescale(scan_im)  # Rescale to 256x256

    # Display two images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(prepare_display(scan_im), caption='Original Image.', width='stretch')

    with col2:
        suppressed_image, ssim, psnr = generate_suppressed_image(scan_im)
        st.image(suppressed_image, caption=f'Fat Suppressed Image. SSIM: {ssim:.2f}, PSNR: {psnr:.2f}', width='stretch')
else:
    st.write("Please upload a DICOM file.")
