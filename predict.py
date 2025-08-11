from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("fruit_quality_model.h5")
print("Model loaded successfully!")

class_indices = {'FreshApple': 0, 'FreshMango': 1, 'FreshOrange': 2, 'FreshPotato': 3, 'FreshTomato': 4,
                 'RottenApple': 5, 'RottenMango': 6, 'RottenOrange': 7, 'RottenPotato': 8, 'RottenTomato': 9}

# Sort class names by their assigned index to get the class_labels list
class_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]


img_path = "/Users/VaibhavUrs/Downloads/s23.jpg"
img = image.load_img(img_path, target_size=(64, 64))

# Convert to numpy array
img_array = image.img_to_array(img)        
img_array = img_array / 255.0              
img_array = np.expand_dims(img_array, 0)   

predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_index]
confidence = round(100 * predictions[0][predicted_index], 2)

print(f"🖼 Image: {img_path}")
print(f"🔮 Prediction: {predicted_label} ({confidence}% confidence)")

import matplotlib.pyplot as plt
img_plot = image.load_img(img_path)
plt.imshow(img_plot)
plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
