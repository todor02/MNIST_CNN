import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import os

# Load the saved model
model = tf.keras.models.load_model('keras_model.keras')

def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28,28))
    # Convert RGB to grayscale
    img = img.convert('L')
    # Invert image colors
    img = ImageOps.invert(img)
    # Normalize pixel values
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    # Predict digit
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white', cursor='cross')
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 16))
        self.classify_btn = tk.Button(self, text="Predict", command=self.classify_handwriting)
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear_all)
        self.canvas.grid(row=0, column=0, pady=2, padx=2)
        self.label.grid(row=0, column=1, padx=2, pady=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=0, pady=2)
        self.image1 = PIL.Image.new("RGB", (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill='white')

    def classify_handwriting(self):
        # Save the canvas image
        filename = "temp.png"
        self.image1.save(filename)
        # Open the image
        img = Image.open(filename)
        # Predict the digit
        digit, acc = predict_digit(img)
        acc = round(float(acc) * 100, 2)
        self.label.configure(text=f"Digit: {digit}, Confidence: {acc}%")
        os.remove(filename)

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

# Run the application
app = App()
mainloop()
