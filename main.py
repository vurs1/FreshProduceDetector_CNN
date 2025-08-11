from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 64, 64
batch_size = 32

train_dir = "../Visual_Dataset/Train"
val_dir = "../Visual_Dataset/Validation"
test_dir = "../Visual_Dataset/Test"

train_datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    horizontal_flip=True     
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # multi-class classification
)

print(train_generator.class_indices)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

num_classes = len(train_generator.class_indices)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())


epochs = 20


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

#save model
model.save('fruit_quality_model.h5')
print("Model saved to 'fruit_quality_model.h5'")








