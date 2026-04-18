# Sign Language Recognition

A comprehensive Computer Vision and Machine Learning project for recognizing American Sign Language (ASL) using OpenCV, MediaPipe, and trained deep learning models. This project supports both **static letter recognition** (single frames) and **dynamic word recognition** (temporal sequences with LSTM).

## Features

- 🔤 **Static Letter Recognition** - Classify individual ASL letters from your own webcam data using Random Forest
- 🎬 **Dynamic Word Recognition** - Recognize sign language words and sequences using LSTM neural networks
- 🖐️ **Single & Two-Hand Detection** - Support for one-handed and two-handed sign recognition
- 📹 **Custom Data Collection** - Collect your own training data directly from your webcam
- 🎯 **Pre-trained Models** - Ready-to-use models included for immediate inference
- 🛠️ **Complete Training Pipeline** - Scripts to collect, train, and evaluate your custom models

## Project Structure

```
sign-language-project/
├── Static_Letters_Recognition/           # Single-frame letter classification
│   ├── model.p                            # Trained Random Forest model
│   ├── collect_imgs.py                    # Capture training images from webcam
│   ├── create_dataset.py                  # Convert images to feature vectors
│   ├── train_classifier.py                # Train Random Forest classifier
│   └── inference.py                       # Real-time letter recognition
│
├── Dynamic_Words_Recognition/             # Multi-frame sequence recognition
│   ├── action_recognition_model.h5        # LSTM model (single hand)
│   ├── collect_sequences.py               # Capture hand pose sequences
│   ├── train_model.py                     # Train LSTM model
│   ├── inference_classifier.py            # Real-time word recognition
│   └── twoHands/
│       ├── action_model_2hands.h5         # LSTM model (two hands)
│       ├── train.py                       # Two-hand model training
│       └── SignLanguageInference.py       # Two-hand inference
│
└── README.md                              # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam for real-time inference
- ~2GB disk space for dataset (optional, for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sign-language-project.git
   cd sign-language-project
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install opencv-python mediapipe scikit-learn tensorflow numpy pandas
   ```

### Usage

**For Static Letter Recognition:**
```bash
cd Static_Letters_Recognition

# Collect training images (required for training)
python collect_imgs.py

# Create dataset from collected images
python create_dataset.py

# Train the model
python train_classifier.py

# Run real-time inference
python inference.py
```

**For Dynamic Word Recognition:**
```bash
cd Dynamic_Words_Recognition

# Collect hand pose sequences (required for training)
python collect_sequences.py

# Train LSTM model
python train_model.py

# Run real-time inference
python inference_classifier.py

# For two-hand recognition:
cd twoHands
python train.py
python SignLanguageInference.py
```

## Data Collection

### For Pre-trained Models (Inference Only)
No additional setup needed! The pre-trained models are included and ready to use for immediate inference.

### For Training Your Own Models

You'll need to collect your own training data using your webcam:

**Static Letter Recognition Data:**
```bash
cd Static_Letters_Recognition
python collect_imgs.py
```
- You'll be prompted to enter labels (e.g., "A", "B", "C")
- Press 'q' to start recording each letter
- 100 images per letter will be automatically captured and saved to `data/` folder

**Dynamic Word Recognition Data:**
```bash
cd Dynamic_Words_Recognition
python collect_sequences.py
```
- Captures 30 sequences per action/word (configurable)
- Each sequence is 30 frames (~1 second of motion)
- Hand landmarks are automatically extracted using MediaPipe
- Data is saved to `MP_Data/` folder

The collected data folders are git-ignored to keep the repository lightweight.

## Models

| Model | Type | Purpose |
|-------|------|---------|
| `model.p` | Random Forest | Static letter classification |
| `action_recognition_model.h5` | LSTM | Single-hand word/sequence recognition |
| `action_model_2hands.h5` | LSTM | Two-hand word/sequence recognition |

## Technologies

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Scikit-learn (Random Forest)
- **Deep Learning**: TensorFlow, Keras (LSTM)
- **Data Processing**: NumPy, Pandas

## File Paths

All scripts use **absolute paths** for data directories to avoid issues with working directory. The data folder is created relative to each script's location:

```python
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
```

This ensures consistent behavior regardless of where you run the command from.

## Performance

- **Static Recognition**: Real-time (25+ FPS on standard hardware)
- **Dynamic Recognition**: Real-time with LSTM inference
- **Accuracy**: Varies by dataset and model (80-95% typical for well-trained models)

## Troubleshooting

**Issue**: Camera not detected
- Solution: Check that your webcam is connected and not in use by other applications

**Issue**: `data/` folder not found
- Solution: Create it manually or run `collect_imgs.py` to auto-create it

**Issue**: ModelNotFound or import errors
- Solution: Ensure virtual environment is activated and all dependencies are installed

## Attribution

- **Hand Tracking**: [MediaPipe by Google](https://mediapipe.dev)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org), [TensorFlow/Keras](https://www.tensorflow.org)

---

