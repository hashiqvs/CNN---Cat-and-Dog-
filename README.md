## CNN Cat & Dog Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either **cats** or **dogs**. This project covers the full deep learning pipeline — from data preparation to real-world inference.

---

##  Project Structure

```
CNN_Cat_and_Dog.ipynb       # Main Jupyter notebook
cats_dogs_cnn.h5            # Saved trained model
```

---

##  Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Core language |
| TensorFlow / Keras | 2.x | Model building & training |
| NumPy | Latest | Array operations |
| Matplotlib | Latest | Plotting accuracy & loss |

---

##  Dataset

- **Source:** Local folder at `/training_set/` with subfolders `cats/` and `dogs/`
- **Total Images:** 8,005 images
  -  Training set: **6,404 images**
  -  Validation set: **1,601 images**
- **Image Size:** Resized to `64 × 64` pixels
- **Classes:** Binary — `Cat (0)` and `Dog (1)`

---

##  Data Preprocessing

Images are loaded and augmented using Keras `ImageDataGenerator`:

| Technique | Value |
|-----------|-------|
| Rescaling (normalization) | `1/255` — pixels scaled to [0, 1] |
| Shear Range | 0.2 |
| Zoom Range | 0.2 |
| Horizontal Flip | Enabled |
| Train/Validation Split | 80% / 20% |

---

##  Model Architecture

```
Input: (64, 64, 3)  ← RGB image
│
├── Conv2D (32 filters, 3×3, ReLU)
├── MaxPooling2D (pool=2×2, stride=2)
│
├── Conv2D (32 filters, 3×3, ReLU)
├── MaxPooling2D (pool=2×2, stride=2)
│
├── Flatten
├── Dense (128 units, ReLU)
├── Dropout (0.5)
│
└── Dense (1 unit, Sigmoid)  ← Output probability
```

**Classification rule:**
- `output > 0.5` →  **Dog**
- `output ≤ 0.5` →  **Cat**

---

##  Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 32 |

---

##  Training Results

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|---------------|-------------|------------|----------|
| 1 | 56.2% | 61.4% | 0.6864 | 0.6559 |
| 2 | 61.9% | 64.5% | 0.6565 | 0.6302 |
| 3 | 66.7% | 68.6% | 0.6202 | 0.6005 |
| 4 | 69.9% | 71.3% | 0.5798 | 0.5560 |
| 5 | 72.2% | 69.8% | 0.5533 | 0.5613 |
| 6 | 73.1% | 72.5% | 0.5368 | 0.5380 |
| 7 | 75.1% | 75.3% | 0.5145 | 0.5032 |
| 8 | 75.3% | 74.9% | 0.5019 | 0.4965 |
| 9 | 76.7% | 76.8% | 0.4780 | 0.4745 |
| **10** | **77.9%** | **76.3%** | **0.4704** | **0.4849** |

**Final Validation Accuracy: ~76%**

---

##  Inference

To predict on a new image:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
test_image = image.load_img('your_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

# Predict
result = cnn.predict(test_image)

if result[0][0] > 0.5:
    print("Dog ")
else:
    print("Cat ")
```

###  Test Results

| Image | Prediction | Result |
|-------|-----------|--------|
| `cat images.jpeg` | Cat |  Correct |
| `Dog image .jpg` | Dog |  Correct |

---

##  Saving & Loading the Model

**Save:**
```python
cnn.save("cats_dogs_cnn.h5")
```

**Load:**
```python
from tensorflow.keras.models import load_model
model = load_model("cats_dogs_cnn.h5")
```

>  **Note:** The `.h5` format is considered legacy. It is recommended to use the native Keras format instead:
> ```python
> cnn.save("cats_dogs_cnn.keras")
> ```

---

##  Key Insights

###  Strengths
- Clean, modular notebook structure
- Dropout (0.5) effectively reduces overfitting
- Data augmentation improves generalization
- Validation accuracy closely tracks training accuracy — no significant overfitting

###  Potential Improvements

| Issue | Recommendation |
|-------|---------------|
| Small image size (64×64) | Increase to 128×128 or 224×224 |
| Fixed 10 epochs | Add Early Stopping callback |
| No LR scheduler | Use `ReduceLROnPlateau` |
| Hardcoded file paths | Use relative paths or config file |
| No separate test set | Create a dedicated holdout test set |
| Custom CNN (~76%) | Try Transfer Learning (MobileNetV2, VGG16) for 95%+ accuracy |

---

##  Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Run the Notebook

```bash
jupyter notebook CNN_Cat_and_Dog.ipynb
```

---

##  License

This project is for educational purposes.
