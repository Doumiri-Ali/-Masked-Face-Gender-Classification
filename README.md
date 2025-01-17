# Masked Face Gender Classification Project

This project implements a Convolutional Neural Network (CNN) for gender classification of people wearing face masks. The system is designed to accurately determine gender even when facial features are partially obscured by masks, making it particularly useful in healthcare settings, public spaces, and other scenarios where face masks are common.

## Project Structure

```
gender-classification/
├── model.py           # Model training script for masked face classification
├── app.py            # Streamlit web application
├── requirements.txt  # Project dependencies
├── Training/        # Training dataset directory
│   ├── Male/       # Male images with masks
│   └── Female/     # Female images with masks
└── Validation/     # Validation dataset directory
    ├── Male/       # Male images with masks
    └── Female/     # Female images with masks
```

## Key Features

- Specialized for gender classification of masked faces
- Works with various types of face masks (medical, cloth, N95)
- Real-time processing through webcam
- Batch processing for uploaded images
- High accuracy despite partial face occlusion
- User-friendly web interface

## Model Architecture

The model uses a CNN architecture specifically optimized for masked face classification:

1. Data Augmentation:
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

2. CNN Layers:
```python
model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])
```

## Training Process

The model is trained on a dataset of masked face images using:
- Input image size: 96x96 pixels
- Batch size: 20
- Epochs: 17
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

Training code:
```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=17
)
```

## Web Application Features

The Streamlit app (`app.py`) provides:

1. Two input methods:
   - Image upload
   - Webcam capture

2. Real-time processing:
   - Automatic image preprocessing
   - Gender prediction with confidence score

3. User-friendly interface:
   - Simple tab-based navigation
   - Progress indicators
   - Clear prediction display

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Use the interface:
   - Choose "Upload Image" or "Webcam" tab
   - Follow on-screen instructions
   - Get gender predictions with confidence scores

## Dependencies

```
tensorflow-macos==2.12.0
tensorflow-metal==1.0.0
streamlit==1.31.1
opencv-python-headless==4.9.0.80
Pillow==10.2.0
numpy==1.24.3
```

## Model Details

The model architecture is optimized for masked face analysis:
- Input normalization (rescaling pixel values to [0,1])
- 4 convolutional blocks with increasing filters (16→32→64→128)
- MaxPooling layers for dimensionality reduction
- Dense layers for final classification
- Dropout for regularization
- Focus on upper face features (eyes, eyebrows, forehead)
- Robust to various mask types and positions

Data augmentation helps handle variations in:
- Mask types and colors
- Mask wearing styles
- Face angles and positions
- Lighting conditions

The model saves both the full model and weights:
- Full model: `gender_classification_model.keras`
- Weights only: `gender_classification_weights.h5` 
