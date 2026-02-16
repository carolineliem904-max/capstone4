# 🦺 Construction Safety PPE Detection System

An automated computer vision system that detects construction workers and evaluates Personal Protective Equipment (PPE) compliance using YOLOv8. The system identifies persons, helmets, and safety vests, then calculates real-time safety compliance scores.

**Key Features:**
- 👷 Person detection
- 🪖 Helmet compliance monitoring
- 🦺 Safety vest verification
- 📊 Automated safety scoring

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd construction-ppe-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Basic Usage

1. **Upload Images**: Drag and drop or browse for construction site images
2. **Adjust Settings**: Configure detection parameters in the sidebar
3. **View Results**: Get annotated images with compliance reports
4. **Download Reports**: Export results as CSV and annotated images

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Model Details](#-model-details)
- [How It Works](#-how-it-works)
- [Application Features](#️-application-features)
- [Model Performance](#-model-performance)
- [Configuration Guide](#-configuration-guide)
- [Training Your Own Model](#-training-your-own-model)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Project Overview

This project provides an automated PPE compliance monitoring system for construction sites using deep learning object detection. The system helps safety managers and site supervisors ensure worker safety by automatically detecting PPE violations.

### Workflow

1. **Detection**: Identifies persons and their PPE items in images
2. **Association**: Links helmets and vests to detected persons using spatial analysis
3. **Evaluation**: Determines compliance based on regional detection logic
4. **Reporting**: Generates safety scores and detailed compliance reports

### Detection Classes

- `person` - Construction worker
- `helmet` - Safety helmet (compliant)
- `no-helmet` - Missing helmet (violation)
- `vest` - Safety vest (compliant)
- `no-vest` - Missing vest (violation)

---

## 🧠 Model Details

**Architecture:** YOLOv8s (Small variant)
- **Model Size**: 22.5 MB
- **Input Resolution**: 768×768 pixels
- **Framework**: Ultralytics YOLOv8 v8.4.14
- **Training Duration**: 33 minutes on Tesla T4 GPU
- **Inference Speed**: ~75 FPS (13.4ms per image)

### Training Configuration

```yaml
Base Model: YOLOv8s pretrained
Image Size: 768×768
Epochs: 80 (early stopped at 64)
Best Checkpoint: Epoch 44
Optimizer: SGD with momentum
Augmentation: Standard YOLOv8 (mosaic, mixup, HSV)
Hardware: Tesla T4 GPU (CUDA)
```

### Dataset Statistics

| Class | Images | Instances |
|-------|--------|-----------|
| Person | 115 | 241 |
| Helmet | 117 | 232 |
| Vest | 74 | 141 |
| No-vest | 52 | 90 |
| No-helmet | 6 | 11 |
| **Total** | **119** | **715** |

> ⚠️ **Note**: Limited no-helmet samples may affect detection robustness for this class.

---

## 🔍 How It Works

### Detection Pipeline

1. **Person Detection**: YOLOv8 identifies all persons in the image
2. **Size Filtering**: Filters persons below minimum area threshold (default: 2,500 px²)
3. **PPE Detection**: Detects helmets and vests independently
4. **Regional Association**: 
   - **Helmet Zone**: Top 40% of person bounding box (y: 0-40%)
   - **Vest Zone**: Torso region (y: 40-95%)
5. **Conflict Resolution**: Highest confidence wins when both positive/negative classes detected
6. **Compliance Evaluation**: Person passes only if **both** helmet AND vest detected

### Safety Score Formula

```
Safety Score = (Number of Compliant Persons / Total Persons) × 100
```

### Compliance Logic

- ✅ **PASS**: Both helmet AND vest detected in correct regions
- ❌ **FAIL**: Missing helmet OR vest, or explicit negative detection
- ❓ **Unknown**: PPE item not detected (treated conservatively as non-compliant)

---

## 🖥️ Application Features

### Two Input Methods

**1. Upload Images Tab**
- Batch upload multiple images (drag-and-drop supported)
- Process local files from your device
- Download annotated images and CSV reports

**2. Image URL Tab**
- Paste direct image URLs
- Instant analysis from web sources
- No file upload required

### Interactive Settings (Sidebar)

| Setting | Range | Default | Purpose |
|---------|-------|---------|---------|
| **Confidence Threshold** | 0.05-0.90 | 0.35 | Minimum detection confidence |
| **NMS IoU** | 0.10-0.90 | 0.60 | Overlapping detection filter |
| **Inference Size** | 640/768/896 | 768 | Processing resolution |
| **Min Person Area** | 0-20,000 px² | 2,500 | Filter small/distant persons |

### Output Reports

- **Annotated Images**: Color-coded bounding boxes (green=pass, red=fail)
- **Summary Metrics**: Overall safety score, pass/fail counts
- **Per-Person Table**: Individual compliance status with confidence scores
- **CSV Export**: Detailed results for further analysis

---

## 📊 Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Precision** | 87.2% |
| **Recall** | 85.0% |
| **mAP@0.5** | 91.1% |
| **mAP@0.5:0.95** | 51.5% |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| **Helmet** | 90.6% | 94.8% | 95.4% | 53.5% |
| **No-helmet** | 100% | 68.6% | 92.2% | 40.2% |
| **Vest** | 84.9% | 83.0% | 91.0% | 53.1% |
| **No-vest** | 73.1% | 83.3% | 81.1% | 44.3% |
| **Person** | 87.4% | 95.0% | 95.6% | 66.6% |

### Model Strengths ✅

- Excellent helmet detection (95.4% mAP@0.5)
- Strong person detection (95.6% mAP@0.5)
- Fast inference speed (~75 FPS)
- High overall precision (87.2%)
- Good recall for safety violations

### Known Limitations ⚠️

- Lower precision on no-vest class (73.1%)
- Challenges with occluded/partially hidden workers
- Reduced accuracy for very distant workers
- Limited no-helmet training samples (11 instances)
- Performance degrades in poor lighting conditions

---

## ⚙️ Configuration Guide

### Recommended Settings by Use Case

#### 🏗️ Crowded Construction Sites
```python
Confidence Threshold: 0.40-0.50  # Reduce false positives
Min Person Area: 5,000-8,000 px² # Filter distant workers
Image Size: 768                  # Balanced performance
NMS IoU: 0.60                    # Standard overlap
```

#### 👷 Close-up Worker Monitoring
```python
Confidence Threshold: 0.30-0.35  # Catch more detections
Min Person Area: 1,000-2,000 px² # Include nearby workers
Image Size: 768                  # Optimal resolution
NMS IoU: 0.60                    # Standard overlap
```

#### 🎯 Maximum Accuracy (Slower)
```python
Confidence Threshold: 0.35       # Balanced threshold
Min Person Area: 2,500 px²       # Standard filtering
Image Size: 896                  # Highest resolution
NMS IoU: 0.60                    # Standard overlap
# Expected: ~95% mAP@0.5 for helmet/person
```

#### ⚡ Maximum Speed (Faster)
```python
Confidence Threshold: 0.40       # Stricter filtering
Min Person Area: 2,500 px²       # Standard filtering
Image Size: 640                  # Fastest inference
NMS IoU: 0.60                    # Standard overlap
# Maintains >85% mAP@0.5 while maximizing speed
```

#### 📋 Safety Compliance Audits
```python
Confidence Threshold: 0.35       # Catch violations
Min Person Area: 2,500 px²       # Standard filtering
Image Size: 768                  # Best balance
NMS IoU: 0.60                    # Standard overlap
# Achieves 85% recall for violations
```

### Understanding Confidence Thresholds

Based on validation performance:

| Class | Recommended Range | Performance |
|-------|-------------------|-------------|
| **Helmet** | 0.35+ | Very reliable (95.4% mAP@0.5) |
| **Person** | 0.35+ | Excellent (95.6% mAP@0.5) |
| **Vest** | 0.35-0.40 | Good (91.0% mAP@0.5) |
| **No-vest** | 0.35-0.45 | Moderate (81.1% mAP@0.5) |
| **No-helmet** | 0.30-0.40 | Use cautiously (limited training data) |

---

## 🎓 Training Your Own Model

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- YOLOv8 training dataset in YOLO format

### Dataset Format

```yaml
# ppe_dataset.yaml
path: ../datasets/construction_ppe
train: images/train
val: images/val

names:
  0: helmet
  1: no-helmet
  2: no-vest
  3: person
  4: vest
```

### Training Script

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8s.pt')

# Train on your dataset
results = model.train(
    data='ppe_dataset.yaml',
    epochs=80,
    imgsz=768,
    batch=16,
    patience=20,          # Early stopping patience
    name='ppe_detector',
    device=0,             # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=8,
    save=True,
    save_period=10,       # Save checkpoint every 10 epochs
    val=True
)

# Validate final model
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

# Export best model
# Best weights saved at: runs/detect/ppe_detector/weights/best.pt
# Copy to project: cp runs/detect/ppe_detector/weights/best.pt models/
```

### Training Tips

1. **Image Resolution**: 768px provides best balance (tested: 640, 768, 896)
2. **Augmentation**: Default YOLOv8 augmentation works well
3. **Early Stopping**: Use patience=20 to prevent overfitting
4. **Class Balance**: Collect more no-helmet samples if possible
5. **Validation**: Use separate validation set (20-30% of data)

### Experiment Results

| Image Size | Augmentation | Result |
|------------|--------------|--------|
| 640 | Default | Good baseline |
| **768** | **Default** | **Best overall (recommended)** |
| 896 | Default | Marginal improvement, slower |
| 768 | HSV only | Slight performance drop |

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### ❌ Model Not Found
```
Error: Model file not found at 'models/best.pt'
```
**Solution**: 
- Ensure `best.pt` is in the `models/` directory
- Check file permissions
- Download pre-trained model if missing

#### 🎯 Low Detection Accuracy
**Symptoms**: Missing valid detections or too many false positives

**Solutions**:
- **Missing detections**: Lower confidence threshold (try 0.30-0.35)
- **False positives**: Raise confidence threshold (try 0.40-0.50)
- **Background crowd**: Increase min_person_area (try 5,000-8,000 px²)
- **Small objects**: Increase inference size to 896

#### 🌐 URL Image Loading Fails
**Symptoms**: Cannot load images from URLs

**Solutions**:
- Verify URL points directly to image file (ends with .jpg, .png, etc.)
- Check if URL requires authentication
- Try downloading image manually and uploading instead
- Ensure stable internet connection

#### 💾 Memory Issues
**Symptoms**: Out of memory errors with large images

**Solutions**:
- Reduce inference image size to 640
- Process fewer images simultaneously
- Reduce batch size in settings
- Use GPU if available (faster and more memory-efficient)

#### 🐌 Slow Performance
**Symptoms**: Long processing times

**Solutions**:
- Reduce inference size to 640
- Increase confidence threshold to 0.45+
- Reduce image resolution before upload
- Enable GPU acceleration if available

---

## 🧪 Advanced Configuration

### Custom PPE Association Rules

The default regional association can be adjusted by modifying the detection logic:

```python
# Default zones
HELMET_ZONE = (0.0, 0.40)  # Top 40% of person box
VEST_ZONE = (0.40, 0.95)   # Middle to bottom region

# For hard hats that sit higher:
HELMET_ZONE = (0.0, 0.35)  # Top 35%

# For high-visibility vests:
VEST_ZONE = (0.35, 0.90)   # Adjusted torso region
```

### Batch Processing

For processing multiple images efficiently:

```python
import glob
from pathlib import Path

# Process all images in directory
image_dir = Path("construction_images/")
images = list(image_dir.glob("*.jpg"))

for img_path in images:
    # Your detection code here
    results = model.predict(img_path, imgsz=768, conf=0.35)
    # Save results
```

---

## 🚀 Future Improvements

### Planned Features

- [ ] Two-stage detection pipeline (person detection → PPE classification)
- [ ] Real-time CCTV/video stream support
- [ ] Multi-camera synchronization
- [ ] Historical compliance tracking and analytics
- [ ] Mobile app for on-site inspections
- [ ] Additional PPE types (gloves, boots, goggles, harnesses)
- [ ] Night/low-light detection optimization
- [ ] Integration with safety management systems

### Model Enhancements

- [ ] Address class imbalance (especially no-helmet samples)
- [ ] Selective augmentation for rare classes
- [ ] Multi-scale training for better small object detection
- [ ] Ensemble methods for improved accuracy
- [ ] Temporal consistency for video streams
- [ ] Occlusion handling improvements

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Data Collection**: More diverse construction site images, especially:
   - No-helmet violations
   - Various lighting conditions
   - Different vest colors and styles
   - International construction standards

2. **Model Improvements**:
   - Better occlusion handling
   - Multi-scale detection
   - Additional PPE types
   - Real-time optimization

3. **Application Features**:
   - Multi-language support
   - Custom report templates
   - API endpoints
   - Mobile app development


## 🙏 Acknowledgments

### Technology Stack

- **Deep Learning**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- **Framework**: [PyTorch](https://pytorch.org/) 2.9.0
- **Web Interface**: [Streamlit](https://streamlit.io/) 1.28+
- **Computer Vision**: [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

### Infrastructure

- **Dataset Platform**: [Roboflow](https://roboflow.com/) - Dataset preparation and augmentation
- **Training Hardware**: Tesla T4 GPU via Google Colab/Kaggle
- **Cloud Platform**: [Streamlit Cloud](https://streamlit.io/cloud) for deployment

### Community

- Construction safety community for datasets and domain expertise
- Open-source contributors and testers
- YOLOv8 community for model development support

---

## ⚠️ Important Disclaimers

### Safety Notice

**This tool is designed to assist with PPE compliance monitoring but should NOT replace:**
- Human supervision and judgment
- Qualified safety inspectors
- Established safety protocols
- Regular safety training programs
- Local safety regulations and standards

### Usage Guidelines

- Always verify automated detections with manual inspection
- Use as a supplementary tool, not sole compliance method
- Follow your organization's safety policies and procedures
- Comply with local construction safety regulations
- Respect worker privacy and data protection laws

### Limitations

- System accuracy depends on image quality and conditions
- Not suitable as sole evidence for safety violations
- May have reduced accuracy in extreme conditions
- Requires periodic retraining and validation
- Regional safety standards may vary

---

**Built with ❤️ for construction worker safety**
