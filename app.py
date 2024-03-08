import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json

config = json.load(open("model_config.json", "r"))
model: tf.keras.models.Sequential = tf.keras.models.load_model("flower_classifier")
image_height = image_width = 180

window = tk.Tk()

current_image: Image = None
width = 800
height = int(width * 9/16)
window.geometry(f"{width}x{height}")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = int((screen_width/2) - (width/2))
y = int((screen_height/2) - (height/2))
window.geometry(f"+{x}+{y}")

window.resizable(False, False)

left_frame = tk.Frame(window, width=int(width/2), height=height)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

image_label = tk.Label(left_frame)
image_label.pack(side=tk.TOP, padx=20, pady=20)

def select_image():
    global current_image
    file_path = filedialog.askopenfilename()
    current_image = Image.open(file_path)
    current_image = current_image.resize((int(width/2), int(height/2)))
    image = ImageTk.PhotoImage(current_image)
    image_label.config(image=image)
    image_label.image = image
select_button = tk.Button(left_frame, text="Select Image", command=select_image)
select_button.pack(side=tk.BOTTOM, padx=20, pady=20)

right_frame = tk.Frame(window, width=int(width/2), height=height, padx=20, pady=20)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
tk.Label(right_frame, text="Available Classes").pack(side=tk.TOP)
ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 15))

text_labels = config["classes"]
text_vars = []
for label in text_labels:
    label_frame = tk.Frame(right_frame)
    label_frame.pack(side=tk.TOP, padx=5)
    label_text = tk.Label(label_frame, text=label)
    label_text.pack(side=tk.LEFT)

ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(30,0))
tk.Label(right_frame, text="Predicted Class", pady=10).pack(side=tk.TOP)
predicted_label = tk.Label(right_frame, text="No Predictions Yes")
predicted_label.pack(side=tk.TOP, pady=(5, 0))
ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 15))

def predict():
    global predicted_label
    img = current_image
    img = img.resize((image_width, image_height))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = text_labels[np.argmax(predictions)]
    predicted_label.config(text=predicted_class)
predict_button = tk.Button(right_frame, text="Predict", command=predict)
predict_button.pack(side=tk.BOTTOM, padx=20, pady=20)

window.mainloop()