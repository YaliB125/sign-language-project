# Sign Language Recognition

A comprehensive Computer Vision and Machine Learning project for recognizing sign language using OpenCV, MediaPipe, and trained deep learning models.

## Project Structure

```
├── Static_Letters_Recognition/      # Single-image letter classification
│   ├── model.p                       # Trained Random Forest model
│   ├── train_classifier.py           # Model training script
│   ├── create_dataset.py             # Dataset preprocessing
│   ├── collect_imgs.py               # Image data collection
│   └── inference.py                  # Real-time inference
│
├── Dynamic_Words_Recognition/        # Multi-frame word recognition with LSTM
│   ├── action_recognition_model.h5   # Trained LSTM model (single hand)
│   ├── twoHands/
│   │   ├── action_model_2hands.h5    # Trained LSTM model (two hands)
│   │   ├── train.py                  # Training script for two-hand model
│   │   └── SignLanguageInference.py  # Inference with two hands
│   ├── train_model.py                # LSTM model training
│   ├── collect_sequences.py          # Temporal data collection
│   └── inference_classifier.py       # Real-time inference
│
└── test.py                           # General testing script
```

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sign-language-recognition.git
cd sign-language-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe scikit-learn tensorflow numpy
```

### 2. Download the Dataset

**You need to download the ASL (American Sign Language) dataset separately:**

1. Go to [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Download the dataset (you may need a Kaggle account)
3. Extract it to your local machine

### 3. Place Dataset in Project

Create the following folder structure and place your data:

```
sign-language-recognition/
├── data/
│   ├── asl_alphabet_train/    # Training images (place extracted Kaggle data here)
│   └── asl_alphabet_test/     # Test images (optional)
```

**The `data/` folder is git-ignored and will not be tracked in version control.** This keeps the repository lightweight while allowing scripts to find the data locally.

### 4. Running the Project

**For Static Letter Recognition:**
```bash
cd Static_Letters_Recognition

# Collect new training images (optional)
python collect_imgs.py

# Create dataset from images
python create_dataset.py

# Train the model
python train_classifier.py

# Run inference on webcam
python inference.py
```

**For Dynamic Word Recognition:**
```bash
cd Dynamic_Words_Recognition

# Collect sequence data (optional)
python collect_sequences.py

# Train LSTM model
python train_model.py

# Run inference
python inference_classifier.py

# For two-hand recognition:
cd twoHands
python train.py
python SignLanguageInference.py
```

## Models Included

- **`model.p`** - Pre-trained Random Forest classifier for static letter recognition
- **`action_recognition_model.h5`** - Pre-trained LSTM model for single-hand dynamic word recognition
- **`action_model_2hands.h5`** - Pre-trained LSTM model for two-hand dynamic word recognition

These models are included in the repository and ready to use for inference.

## Technologies Used

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Scikit-learn (Random Forest)
- **Deep Learning**: TensorFlow, Keras (LSTM)
- **Data Processing**: NumPy, Pandas

## Dataset Attribution

The static letter recognition uses the **ASL Alphabet dataset** from Kaggle:
- [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Dataset contains 87,000+ images of American Sign Language letters

## Contributing

Feel free to fork, modify, and submit improvements!

## License

[Add your license here - MIT, Apache 2.0, etc.]

## Author

[Your Name]

---

**Note**: Make sure to download the dataset as described above before running training or inference scripts. The `data/` folder is not included in version control due to its size.
