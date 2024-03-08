import os
import tensorflow as tf
import json
os.system("cls")

image_height = image_width = 180
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

classes = train_ds.class_names

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  epochs=epochs
)

model.save("flower_classifier")
with open("model_config.json", "w") as f:
    json.dump({
        "classes": classes
    }, f, indent=1)
