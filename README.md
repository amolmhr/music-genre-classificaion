# GTZAN Music Genre Classification

## Overview
This repository contains code and resources for classifying music genres using the GTZAN dataset. The GTZAN dataset is a widely-used benchmark in music information retrieval, containing 1,000 audio tracks categorized into 10 different genres.

## Dataset
The GTZAN dataset consists of 1,000 audio files, each 30 seconds long, categorized into the following genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

### Dataset Source
The dataset can be downloaded from [Marsyas](http://marsyas.info/downloads/datasets.html). Ensure you follow the licensing terms provided by the dataset creators.

## Requirements
To run the code, you need the following:

### Python Libraries
- Python 3.8+
- NumPy
- pandas
- librosa
- scikit-learn
- TensorFlow or PyTorch (depending on the chosen framework)
- Matplotlib
- seaborn

Install the required packages using the command:
```bash
pip install -r requirements.txt
```

## Features and Preprocessing
The project includes the following preprocessing steps:
1. **Feature Extraction:**
   - Mel-frequency cepstral coefficients (MFCCs)
   - Spectral Centroid
   - Spectral Rolloff
   - Zero Crossing Rate
   - Chroma Features
2. **Data Augmentation (Optional):**
   - Time Stretching
   - Pitch Shifting
   - Adding Noise

## Model Architecture
The classification models supported in this project include:
1. **Traditional Machine Learning Models:**
   - Support Vector Machines (SVM)
   - Random Forests
   - k-Nearest Neighbors (k-NN)
2. **Deep Learning Models:**
   - Convolutional Neural Networks (CNNs)
   - Recurrent Neural Networks (RNNs)
   - Combined CNN-RNN architectures

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/amolmhr/music-genre-classification.git
   cd gtzan-genre-classification
   ```

2. Extract the dataset into the `data` directory.

3. Run preprocessing:
   ```bash
   python preprocess.py
   ```

4. Train the model:
   ```bash
   python train.py --model cnn
   ```

5. Evaluate the model:
   ```bash
   python evaluate.py --model cnn
   ```

6. Visualize the results:
   ```bash
   python visualize_results.py
   ```

## Results
The model achieves the following accuracies:
- CNN: ~85%
- SVM: ~70%
- Random Forest: ~65%

The performance can vary based on hyperparameter tuning and the use of data augmentation.

## Folder Structure
```
|-- data
|   |-- genres
|-- models
|   |-- cnn.py
|   |-- svm.py
|   |-- rnn.py
|-- preprocess.py
|-- train.py
|-- evaluate.py
|-- visualize_results.py
|-- requirements.txt
|-- README.md
```

## Future Work
- Experiment with transfer learning using pre-trained audio models.
- Implement more advanced augmentation techniques.
- Add multilingual genre classification support.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License
This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- GTZAN dataset creators
- Marsyas framework developers
- Community contributors in music information retrieval

