## Note On Graph Logs
This page contains screenshots of the model training process from my VS Code terminal.  
These logs show how the CNN's accuracy and loss evolved over each epoch during training.


From the very first epoch, the accuracy improved steadily and both training and validation losses decreased.

## Reasons Why Accuracy Improved During Training
- **Gradient-based learning:** The CNN adjusts its weights using backpropagation and gradient descent after each batch, improving predictions gradually.
- **Data augmentation:** Applying random transformations (rotation, zoom, flip) to training images helped the model generalize better to unseen variations.
- **Proper normalization:** Rescaling pixel values to the 0–1 range prevented saturation and helped stable convergence.
- **Batch training:** Updating weights after batches rather than single images led to smoother, more effective learning.
- **Sufficient training epochs:** Multiple epochs gave the model enough time to learn from the dataset without overfitting severely.

## Final Accuracy
After completing training, the model achieved:

**Final Test Accuracy:** 91.74%

Below is an example training run for 20 epochs:

## Graph Logs
<img width="1189" height="689" alt="Image" src="https://github.com/user-attachments/assets/38193fb3-5da6-4563-8a23-f93041ed217b" />
