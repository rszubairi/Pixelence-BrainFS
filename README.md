![image](https://github.com/AmrAMHD/BrainEnhance/assets/170816158/efe79fd8-d3a8-4cb5-8bcb-bb4f410d01f6)
# FatSuppress: AI-powered MRI Fat Suppression (v1.0)

FatSuppression is a Python-based deep learning tool designed to generate synthetic fat-suppressed MRI scans from standard MRI sequences. This approach eliminates the need for traditional fat suppression techniques, offering benefits in scan efficiency, cost reduction, and improved image quality.

## Technical Specifications:
- Backend: Python
- Deep Learning Frameworks: TensorFlow, Keras
- Medical Image Processing Libraries: Pydicom, NumPy
- Visualization Library: Matplotlib

## Hardware Requirements:
- Minimum: 16 GB RAM (preferred 32 GB RAM)
- Recommended: NVIDIA GPU with at least 8 GB VRAM (preferred 12 GB VRAM)

## Core Technology:
FatSuppression utilizes a 2D U-Net architecture with attention blocks, optimized for medical image synthesis. U-Nets are ideal for segmentation and synthesis tasks, and attention mechanisms further enhance performance by focusing on the most informative image features. This leads to high-quality fat suppression with minimal artifacts.

## Potential Benefits:

- Improved Image Quality: Synthetic fat suppression can reduce artifacts seen in conventional suppression techniques.
- Reduced Scan Time: Eliminating dedicated fat suppression sequences can shorten MRI scan duration.
- Enhanced Consistency: Overcomes variability associated with traditional fat suppression methods, especially in challenging cases.
- Safer for Patients: No additional RF pulses or contrast agents required.

## Network Architecture:
The model takes a mono-channel MRI input: T2-weighted. This allows the network to capture essential tissue contrasts. The architecture is based on a U-Net structure with attention-enhanced skip connections, enabling precise fat suppression while preserving anatomical details.

## Example Input and Output:
![Figure 2024-08-29 180807 (76)](https://github.com/user-attachments/assets/315dbd9b-1a19-4680-bfdd-449966de2e30)
![Figure 2024-08-29 194228 (11)](https://github.com/user-attachments/assets/aaf38a24-4b69-4857-bbcf-828a2e52ac84)
![Figure 2024-08-29 180807 (68)](https://github.com/user-attachments/assets/21622f89-bca5-4340-9948-a45b586a3c41)
![Figure 2024-08-29 180807 (69)](https://github.com/user-attachments/assets/aaa51769-1eb9-41cd-b569-52778ffc1140)

# Disclaimer:
FatSuppression is a research tool in development and is not intended for clinical diagnosis at this stage. Further validation and regulatory approval are required before clinical application.

# Future Developments:
- 3D U-Net Implementation: To improve spatial consistency in synthetic fat suppression.
- Multi-modal Learning: Incorporation of additional MRI sequences for enhanced suppression quality.
- Regulatory Compliance: Efforts toward clinical validation and approval.

This documentation provides an overview of FatSuppression v1.0. As the project progresses, updates will reflect advancements and improvements.
** Dr. Amr Muhammed, M.D FRCR, M.Sc. PhD
** amr.muhammed@med.sohag.edu.eg
