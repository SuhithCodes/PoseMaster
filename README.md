# 🧘 PoseMaster: AI-Powered Yoga Position Detection

> An intelligent system that provides real-time yoga pose detection and guidance using computer vision and machine learning.

## 🎯 Overview

PoseMaster is a cutting-edge automated system designed to detect yoga positions using computer vision and machine learning techniques. The system analyzes images and videos in real-time to recognize various yoga poses and provide instant feedback on posture alignment, making yoga practice more accessible and effective for everyone.

## ✨ Key Features

- 🎥 Real-time pose detection and analysis
- 🎯 Instant feedback on posture alignment
- 🤖 Advanced machine learning algorithms
- 📊 High accuracy and performance
- 🔄 Support for multiple yoga poses

## 🎓 Supported Yoga Poses

- 🏋️ Plank
- 🐕 Downdog
- 👑 Goddess
- 💃 Dancers
- ⚔️ Warrior 2

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install via pip)

### Installation

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Processing Pipeline:**

   a. **Data Augmentation** 🔄

   ```bash
   python data_argumentation.py
   ```

   - Horizontal flipping for variety
   - Gaussian noise for resilience
   - Image scaling and standardization
   - Brightness and contrast adjustment

   b. **Feature Extraction** 📊

   ```bash
   python convert_data.py
   ```

   - MediaPipe integration for landmark detection
   - 33 body joint location extraction

   c. **Feature Engineering** ⚙️

   ```bash
   python FE_data.py
   ```

   - Euclidean distance calculations
   - Joint angle measurements
   - Dimensionality reduction

   d. **Data Cleaning** 🧹

   ```bash
   python removing_outliers.py
   ```

   - Z-score based outlier removal

## 🤖 Model Training

Run the training pipeline:

```bash
python main.py
```

### Model Architecture

Our system employs a sophisticated ensemble approach combining:

- 🎯 **Support Vector Machines (SVM)**

  - Effective in high-dimensional spaces
  - Robust against overfitting

- 🌳 **Random Forest**

  - Excellent for large datasets
  - Built-in protection against overfitting

- ⚡ **CATBoost**

  - High performance and scalability
  - Efficient processing

- 🔄 **Stacking with Logistic Regression**
  - Combines model strengths
  - Optimizes prediction accuracy

## 📊 Performance Metrics

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | 98%   |
| Validation Accuracy | 86%   |
| Test Accuracy       | 80%   |

The stacked model significantly outperforms the baseline CNN approach, demonstrating the effectiveness of our methodology.

## 🛠️ Technical Implementation

### Data Pipeline

1. **Data Collection**

   - Curated from "Yoga Poses Dataset" and "Yoga-82"
   - Diverse pose variations and lighting conditions

2. **Preprocessing**

   - Image resizing and normalization
   - Contrast enhancement
   - Ground truth annotation

3. **Feature Engineering**
   - MediaPipe landmark extraction
   - Geometric feature calculation
   - Advanced heuristic implementation

## 🔮 Future Developments

- Expansion of supported yoga poses
- Enhanced real-time performance
- Mobile application development
- Integration with smart fitness devices

## 🌟 Conclusion

PoseMaster represents a significant step forward in making yoga practice more accessible and effective through technology. By providing real-time feedback and guidance, it helps practitioners improve their form and get the most out of their yoga practice.

---

_Built with ❤️ using Python, OpenCV, and MediaPipe_
