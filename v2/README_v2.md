# Fat Suppression Model v2: 3D U-Net with Multi-modal Learning

This directory contains the enhanced v2 implementation of the fat suppression model with the following key improvements:

## 🚀 Key Features

### 1. 3D U-Net Architecture
- **Spatial Consistency**: 3D convolutions maintain spatial relationships across volumetric data
- **Enhanced Attention**: 3D CBAM (Convolutional Block Attention Module) for better feature selection
- **Deeper Architecture**: 4-level encoder-decoder with skip connections

### 2. Multi-modal Learning
- **Fusion Capabilities**: Support for multiple MRI sequences (T1, T2, FLAIR, etc.)
- **Modality-Specific Processing**: Individual attention mechanisms per modality
- **Adaptive Fusion**: Concatenation-based modality fusion at input level

### 3. Advanced Training Features
- **GAN Training Option**: Optional adversarial training for improved generation quality
- **3D Visualization**: Multi-planar reconstruction for training monitoring
- **Enhanced Callbacks**: 3D-specific performance monitoring and visualization

## 📁 File Structure

```
v2/
├── models_v2.py              # 3D U-Net and discriminator models (TensorFlow)
├── models_v2_torch.py        # 3D U-Net and discriminator models (PyTorch)
├── datagen_v2.py             # 3D and multi-modal data generators
├── train_v2.py               # Training script with argument parsing
├── utils_v2.py               # Enhanced utilities for 3D processing
├── visualize_brainFS_v2.py   # 3D visualization web app
├── test_v2.py                # Comprehensive test suite
└── README_v2.md              # This documentation
```

## 🛠️ Installation & Setup

### Requirements
```bash
pip install -r ../requirements.txt
```

### Dependencies
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-Image
- Pandas
- tqdm

## 📊 Usage

### Training a 3D Single-Modality Model
```bash
cd v2
python train_v2.py \
    --data_path ../processed/ \
    --batch_size 1 \
    --epochs 100 \
    --volume_depth 32 \
    --volume_height 128 \
    --volume_width 128 \
    --filters 16 \
    --learning_rate 1e-4
```

### Training a Multi-Modal Model
```bash
python train_v2.py \
    --data_path ../processed/ \
    --num_modalities 2 \
    --batch_size 1 \
    --epochs 100 \
    --volume_depth 32 \
    --volume_height 128 \
    --volume_width 128 \
    --filters 16
```

### Training with GAN
```bash
python train_v2.py \
    --use_gan \
    --data_path ../processed/ \
    --batch_size 1 \
    --epochs 200
```

### Data Generator Sanity Check
```bash
python train_v2.py --sanity_check --data_path ../processed/
```

### Running the 3D Visualization Web App
```bash
cd v2
streamlit run visualize_brainFS_v2.py
```

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | `processed/` | Path to training data |
| `--valid_path` | `processed/` | Path to validation data |
| `--batch_size` | `1` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--learning_rate` | `1e-4` | Optimizer learning rate |
| `--filters` | `16` | Base number of filters |
| `--num_modalities` | `1` | Number of input modalities |
| `--volume_depth` | `32` | 3D volume depth |
| `--volume_height` | `128` | 3D volume height |
| `--volume_width` | `128` | 3D volume width |
| `--use_gan` | `False` | Enable GAN training |
| `--sanity_check` | `False` | Run data generator check |

## 🏗️ Architecture Details

### 3D U-Net Components
- **Encoder**: 4 levels with 3D convolutions, batch normalization, dropout
- **Decoder**: 4 levels with 3D transpose convolutions and skip connections
- **Attention**: 3D CBAM blocks at each level
- **Bottleneck**: Enhanced feature processing with attention

### Multi-modal Fusion
- **Input Processing**: Each modality processed with dedicated attention
- **Fusion Strategy**: Channel-wise concatenation of processed modalities
- **Feature Learning**: Joint learning across modalities

### Loss Functions
- **Primary Loss**: VGG-based perceptual loss
- **Additional Metrics**: MAE, SSIM, PSNR
- **GAN Loss**: Optional adversarial loss for enhanced generation

## 📈 Performance Monitoring

### Training Callbacks
- **Model Checkpoint**: Saves best weights based on validation loss
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduler**: Adaptive learning rate reduction
- **Performance Visualization**: 3D multi-planar reconstruction

### Metrics Tracking
- Training history saved to `model_history_v2.csv`
- Performance plots saved as `training_history_v2.png`
- 3D visualization during training

## 🔍 Data Preparation

### Input Format
The model expects 3D volumes organized as:
```
processed/
├── fat/           # Input volumes
│   ├── vol_001_slice_01.jpg
│   ├── vol_001_slice_02.jpg
│   └── ...
└── suppressed/    # Target volumes
    ├── vol_001_slice_01.jpg
    ├── vol_001_slice_02.jpg
    └── ...
```

### Multi-modal Setup
For multi-modal training, organize data as:
```
processed/
├── modality_0/fat/     # First modality inputs
├── modality_0/suppressed/
├── modality_1/fat/     # Second modality inputs
└── modality_1/suppressed/
```

## 🎯 Expected Improvements

### Spatial Consistency
- Better preservation of 3D anatomical structures
- Improved fat suppression across volumetric slices
- Enhanced temporal/spatial coherence

### Multi-modal Benefits
- Complementary information from different sequences
- Improved robustness to imaging artifacts
- Better feature representation

### Performance Metrics
- Higher SSIM and PSNR scores
- Improved perceptual quality
- Better generalization to unseen data

## 🐛 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or volume dimensions
2. **Data Loading Errors**: Check file paths and naming conventions
3. **Training Instability**: Adjust learning rate or use gradient clipping

### Performance Tips
- Use smaller volumes for initial testing
- Enable mixed precision training for faster computation
- Monitor GPU memory usage during training

## 📝 Future Enhancements

- **Advanced Fusion**: Cross-modal attention mechanisms
- **Temporal Modeling**: 3D+time for dynamic sequences
- **Self-supervised Learning**: Pre-training strategies
- **Domain Adaptation**: Cross-site generalization

## 🤝 Contributing

When contributing to v2 development:
1. Maintain compatibility with existing v1 architecture
2. Add comprehensive documentation
3. Include performance benchmarks
4. Test on multiple datasets

## 📄 License

This implementation follows the same license as the original project.
