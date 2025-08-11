# FreshProduceDetector_CNN
This project detects whether fruits are fresh or rotten using a Convolutional Neural Network (CNN). It identifies different fruit types and their freshness based on a dataset of images that it has been trained upon. 
Currently, the model can classify the freshness of the following produce:
**Apple**, **Mango**, **Orange**, **Potato**, and **Tomato**.

## About This Project
I built a CNN using **TensorFlow** and **Keras** to classify fruit images into specific classes like **FreshApple**, **RottenApple**, etc., and also display the **confidence level** of the prediction.  

I trained the model on a dataset of labeled fruit images, tested it on unseen images, and achieved strong accuracy in predictions.

# Source Code Architecture
FreshProduceDetector_CNN/
├── src/
│ ├── main.py # Training script – builds and trains the CNN
│ ├── predict.py # Script for predicting class from a single image
├── fruit_quality_model.h5 # Saved model file after training
├── requirements.txt # Required Python packages for this project
├── README.md # This file you’re reading now


## How to Set Up and Run

### 1. Create and activate your Python virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate # macOS/Linux

### 2. Install dependencies:
pip install -r requirements.txt


### 3. Train the CNN model:
Go into the `src` folder and run:
python main.py
This will train your CNN and save the model as `fruit_quality_model.h5` in the project root.

### 4. Predict on a new image:
Edit `src/predict.py` to update the `img_path` variable with the path to your test image. Then run:
python predict.py

The script will print the predicted fruit class and confidence.


## Notes
- Model trained on **64x64 pixel** images.
- Pixel values are **normalized (0–1)** during training and prediction.
- Class labels and their order **match exactly** the dataset folder structure used during training.








