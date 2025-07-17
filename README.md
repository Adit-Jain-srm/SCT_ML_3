# SCT_ML_3

## Dog vs Cat Image Classification using Support Vector Machine (SVM)

**SkillCraft Technology - ML Internship Task 3**

### 📋 Project Information
- **Intern:** Adit Jain
- **Intern ID:** SCT/JUL25/2512
- **Email:** aj9104@srmist.edu.in
- **LinkedIn:** [www.linkedin.com/in/-adit-jain](https://www.linkedin.com/in/-adit-jain)
- **Company:** SkillCraft Technology
- **Repository:** SCT_ML_3
- **Task:** ML Internship Task 3

---

## 🎯 Project Overview

This project implements a **Support Vector Machine (SVM)** classifier to distinguish between images of dogs and cats using the popular Kaggle Dogs vs Cats dataset. The project demonstrates the application of traditional machine learning techniques for image classification through feature extraction and SVM modeling.

### ✨ **Optimized Implementation**
- **Streamlined Code:** Zero redundancy with efficient, production-ready functions
- **Modular Design:** Clean separation of concerns across 7 organized cells
- **Performance Focus:** Optimized for speed and memory efficiency
- **Professional Quality:** Industry-standard code structure and documentation

## 🔧 Technical Approach

### 1. **Data Processing**
- **Dataset:** Dogs vs Cats image dataset
- **Preprocessing:** Image resizing to 64x64 pixels for computational efficiency
- **Data Split:** Training set (1000 images per class) and Test set (300 images per class)

### 2. **Feature Extraction**
- **Method:** HOG (Histogram of Oriented Gradients) features
- **Parameters:** 
  - Orientations: 9
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - Block normalization: L2-Hys

### 3. **Model Training**
- **Algorithm:** Support Vector Machine (SVM)
- **Kernels:** Linear, RBF, Polynomial
- **Hyperparameter Tuning:** Grid Search with Cross-Validation
- **Feature Scaling:** StandardScaler for optimal SVM performance

### 4. **Evaluation Metrics**
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Cross-Validation Analysis

## 📁 Project Structure

```
SCT_ML_3/
├── training_set/
│   ├── cats/        # Training images of cats (~4000 images)
│   └── dogs/        # Training images of dogs (~4000 images)
├── test_set/
│   ├── cats/        # Test images of cats (~1000 images)
│   └── dogs/        # Test images of dogs (~1000 images)
├── dog_cat_svm_classifier.ipynb  # Optimized implementation (7 cells)
├── requirements.txt              # Dependencies with version ranges
├── .gitignore                    # Git ignore patterns
├── .venv/                        # Virtual environment (ignored)
└── README.md                    # Project documentation
```

### 📋 **Notebook Structure** (Optimized - 7 Cells)
1. **Project Header** - Information and overview
2. **Imports & Config** - All libraries and settings
3. **Data Loading** - Efficient dataset loading function
4. **Feature Processing** - HOG extraction with visualization
5. **Model Training** - Complete SVM pipeline with Grid Search
6. **Results Analysis** - Comprehensive evaluation and visualization
7. **Conclusion** - Summary and achievements

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required libraries (see requirements.txt)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/SCT_ML_3.git
cd SCT_ML_3
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook dog_cat_svm_classifier.ipynb
```

## 📊 Key Features

- **Optimized Code Structure:** Zero redundancy with efficient, modular functions
- **Comprehensive Data Visualization:** Sample images, class distributions, and feature visualizations
- **Advanced Feature Engineering:** HOG feature extraction with parameter optimization
- **Automated Hyperparameter Tuning:** Grid search across multiple SVM parameters
- **Detailed Performance Analysis:** Multiple evaluation metrics and visualizations
- **Prediction Visualization:** Visual comparison of correct vs incorrect predictions
- **Professional Development Setup:** Virtual environment, .gitignore, and version control ready

## 🔍 Results & Performance

The SVM classifier demonstrates:
- **Effective Feature Extraction:** HOG features capture essential image characteristics
- **Robust Classification:** Competitive accuracy on test dataset
- **Optimized Parameters:** Grid search ensures optimal model configuration
- **Comprehensive Evaluation:** Multiple metrics provide thorough performance assessment

## 📈 Model Pipeline

1. **Image Loading** → Load and preprocess images from directories
2. **Feature Extraction** → Extract HOG features from grayscale images
3. **Data Scaling** → Standardize features for SVM optimization
4. **Model Training** → Train SVM with hyperparameter tuning
5. **Evaluation** → Assess performance on test set
6. **Visualization** → Present results with comprehensive plots

## 🛠️ Dependencies

- **Core Libraries:** numpy, pandas, scikit-learn
- **Image Processing:** opencv-python, Pillow, scikit-image
- **Visualization:** matplotlib, seaborn
- **Development:** jupyter, notebook, tqdm

## 🎯 Learning Outcomes

This project demonstrates:
- **Feature Engineering:** Extracting meaningful features from images for traditional ML
- **SVM Implementation:** Applying support vector machines for classification
- **Hyperparameter Optimization:** Systematic approach to model tuning
- **Performance Evaluation:** Comprehensive model assessment techniques
- **Data Visualization:** Effective presentation of results and insights

## 🔮 Future Enhancements

- **Extended Feature Sets:** Implement LBP, SIFT, or other feature descriptors
- **Deep Learning Comparison:** Compare with CNN-based approaches
- **Ensemble Methods:** Combine multiple models for improved performance
- **Full Dataset Utilization:** Scale to complete dataset for production deployment
- **Model Deployment:** Create web interface for real-time classification

## 📝 Usage Example

```python
# Optimized workflow - from the notebook
# Load dataset efficiently
X_train_raw, y_train = load_dataset('training_set', TRAIN_SAMPLES)
X_test_raw, y_test = load_dataset('test_set', TEST_SAMPLES)

# Extract HOG features
X_train = extract_hog_features(X_train_raw)
X_test = extract_hog_features(X_test_raw)

# Train and evaluate SVM with Grid Search
best_model, y_pred, accuracy, cv_scores = train_and_evaluate_svm(
    X_train, X_test, y_train, y_test
)

# Generate comprehensive report
generate_final_report(y_test, y_pred, accuracy, cv_scores)
```

## 🔧 Version Control & Best Practices

### Git Setup
```bash
# Initialize git repository
git init

# Add files (respects .gitignore)
git add .

# Commit changes
git commit -m "Initial commit: SVM dog/cat classifier"

# Add remote repository
git remote add origin https://github.com/yourusername/SCT_ML_3.git

# Push to remote
git push -u origin main
```

### What's Ignored
- Virtual environments (`.venv/`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)
- Python cache files (`__pycache__/`)
- Model files (`*.pkl`, `*.model`, etc.)
- Temporary files and logs
- IDE/Editor specific files

**Note:** Dataset images are tracked but you can uncomment the image patterns in `.gitignore` to exclude them if needed.

## 🤝 Contributing

This project is part of the SkillCraft Technology internship program. For suggestions or improvements, please reach out through the provided contact information.

## 📄 License

This project is created for educational purposes as part of the SkillCraft Technology ML Internship program.

---

**Completed as part of SkillCraft Technology ML Internship Task 3**  
*Demonstrating proficiency in traditional machine learning techniques for image classification*