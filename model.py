import tensorflow as tf
import numpy as np


BATCH = 20
IMG_WIDTH = 96
IMG_HEIGHT = 96

train_loader = tf.keras.preprocessing.image_dataset_from_directory(
"./Training",
seed=123,
image_size=(IMG_HEIGHT, IMG_WIDTH),
batch_size=BATCH
)

test_loader = tf.keras.preprocessing.image_dataset_from_directory(
"./Validation",
seed=123,
image_size=(IMG_HEIGHT, IMG_WIDTH),
batch_size=BATCH
)

class_names = train_loader.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_loader.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_loader.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)


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
    tf.keras.layers.Dense(len(class_names))
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs = 17
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=epochs
)

loss, accuracy = model.evaluate(test_dataset)
print(f"\nTest accuracy: {accuracy*100:.2f}%")

print("\nPrediction Results on Test Images:")
print("-" * 50)
for images, labels in test_loader.take(1):
    for i in range(9):
        predictions = model.predict(tf.expand_dims(images[i], 0), verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(score)
        actual_class = labels[i]
        confidence = 100 * np.max(score)
        
        print(f"Image {i+1}:")
        print(f"Actual: {class_names[actual_class]}")
        print(f"Predicted: {class_names[predicted_class]} (Confidence: {confidence:.2f}%)")
        print("-" * 30)


print("\nSaving model and weights...")
model.save("gender_classification_model.keras")
model.save_weights("gender_classification_weights.h5")
model.save_weights("gender_classification_weights.weights.h5")
print("Model and weights saved successfully!")