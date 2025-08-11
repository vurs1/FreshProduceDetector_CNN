# FreshProduceDetector_CNN
This project detects whether fruits are fresh or rotten using a Convolutional Neural Network (CNN). It identifies different fruit types and their freshness based on a dataset of images that it has been trained upon. 
Currently, the model can classify the freshness of the following produce:
**Apple**, **Mango**, **Orange**, **Potato**, and **Tomato**.

## Results
<div>
  <img src="https://github.com/user-attachments/assets/1c99c8ea-7b30-46e2-a106-c35b5aed8d56" width="320" alt="Training Progress 1" />
  <img src="https://github.com/user-attachments/assets/a3728988-cfb3-4882-aa0f-7bdec570705e" width="320" alt="Training Progress 2" />
</div>

<div>
  <img src="https://github.com/user-attachments/assets/3c83e0f2-f0a2-431e-b247-b0547fcc80d1" width="320" alt="Training Progress 3" />
  <img src="https://github.com/user-attachments/assets/deca4be3-a4ca-456e-ba74-a1ff9075cce4" width="320" alt="Training Progress 4" />
</div>


## About This Project
I built a CNN using **TensorFlow** and **Keras** to classify fruit images into specific classes like **FreshApple**, **RottenApple**, etc., and also display the **confidence level** of the prediction.  

I trained the model on a dataset of labeled fruit images, tested it on unseen images, and achieved strong accuracy in predictions.

## Source Code Architecture
```
FreshProduceDetector_CNN/
├── src/
│   ├── main.py              # Training script – builds and trains the CNN  
│   ├── predict.py           # Script for predicting class from a single image  
├── fruit_quality_model.h5   # Saved model after training  
├── requirements.txt         # Required Python packages  
├── README.md                # Project description and usage instructions  
├── training_logs.md         # Screenshots and notes from the training process  
```

## Model Architecture
<img width="640" height="299" alt="Image" src="https://github.com/user-attachments/assets/7edd66e5-38db-42bf-a3f9-7781546280da" />

**Key Points**:
- Two convolutional layers with max pooling
- Dense layer for classification (with 10 output classes: each fruit/freshness combination)
- 1.6 million trainable parameters

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

