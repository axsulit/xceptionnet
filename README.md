# DeepFake Detection using XceptionNet

XceptionNet implementation for deepfake detection, trained on the FaceForensics++ dataset. The models are trained on slightly enlarged face crops with a scale factor of 1.3.

## Requirements

- Python 3.8+
- See requirements.txt for all dependencies

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the XceptionNet model on your dataset:
```bash
python xception.py 
    -d <path to training data directory>
    -v <path to validation data directory>
    -t <path to test data directory>
    [-p <path to pretrained weights>]
    [-m <path to existing model to continue training>]
    [-b batch_size]
    [-e num_epochs]
    [-lr learning_rate]
    [-s subset_size]
    [-o output_model_path]
```

Optional arguments:
- `-p`: Path to pretrained Xception weights (default: model.pth)
- `-m`: Path to existing model weights to continue training
- `-b`: Batch size (default: 8)
- `-e`: Number of epochs (default: 100)
- `-lr`: Learning rate (default: 0.00005)
- `-s`: Number of images to use for training (subset for testing)
- `-o`: Output path for trained model (default: xception_df.pth)
- `--cuda`: Enable CUDA for GPU acceleration