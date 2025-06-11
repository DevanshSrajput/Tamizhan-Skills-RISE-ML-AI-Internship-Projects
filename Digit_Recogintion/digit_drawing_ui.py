import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from tensorflow import keras
import cv2
import os

class DigitDrawingApp:
    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("600x500")
        
        # Load the trained model
        self.model = None
        self.load_model(model_path)
        
        # Drawing variables
        self.canvas_size = 280
        self.brush_size = 10
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()
    
    def load_model(self, model_path=None):
        """Load the trained model with multiple fallbacks"""
        model_files = []
        
        if model_path:
            model_files.append(model_path)
        
        # Add default model files
        model_files.extend([
            'mnist_digit_model.h5',
            'best_mnist_model.h5',
            os.path.join(os.getcwd(), 'mnist_digit_model.h5'),
            os.path.join(os.getcwd(), 'best_mnist_model.h5')
        ])
        
        for model_file in model_files:
            try:
                if os.path.exists(model_file):
                    self.model = keras.models.load_model(model_file)
                    print(f"✅ Model loaded successfully from {model_file}!")
                    return
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue
        
        print("❌ Could not load any model. Please train the model first.")
        print("Run: python mnist_digit_recognition.py")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Draw a Digit (0-9)", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Model status
        model_status = "✅ Model Loaded" if self.model else "❌ No Model"
        status_color = "green" if self.model else "red"
        status_label = tk.Label(self.root, text=model_status, 
                               font=("Arial", 10), fg=status_color)
        status_label.pack()
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, 
                               height=self.canvas_size, bg="white", 
                               relief=tk.SUNKEN, border=2)
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10)
        
        # Predict button
        predict_btn = tk.Button(buttons_frame, text="Predict Digit", 
                               command=self.predict_digit, bg="green", 
                               fg="white", font=("Arial", 12, "bold"),
                               state="normal" if self.model else "disabled")
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = tk.Button(buttons_frame, text="Clear Canvas", 
                             command=self.clear_canvas, bg="red", 
                             fg="white", font=("Arial", 12, "bold"))
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Prediction result
        self.result_var = tk.StringVar()
        initial_text = "Draw a digit and click 'Predict'" if self.model else "No model loaded - train model first"
        self.result_var.set(initial_text)
        result_label = tk.Label(self.root, textvariable=self.result_var, 
                               font=("Arial", 14, "bold"), fg="blue")
        result_label.pack(pady=10)
        
        # Confidence scores frame
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Instructions: Draw a digit in the white canvas above\n"
                                    "Click 'Predict' to see the model's prediction",
                               font=("Arial", 10), fg="gray")
        instructions.pack(pady=5)
        
    def start_draw(self, event):
        """Start drawing when mouse button is pressed"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_digit(self, event):
        """Draw on canvas as mouse moves"""
        # Draw on tkinter canvas
        self.canvas.create_oval(event.x - self.brush_size, event.y - self.brush_size,
                               event.x + self.brush_size, event.y + self.brush_size,
                               fill="black", outline="black")
        
        # Draw on PIL image
        self.draw.ellipse([event.x - self.brush_size, event.y - self.brush_size,
                          event.x + self.brush_size, event.y + self.brush_size],
                         fill="black")
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("Draw a digit and click 'Predict'")
        
        # Clear confidence scores
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
    
    def preprocess_image(self):
        """Preprocess the drawn image for model prediction"""
        # Convert to grayscale
        img_gray = self.image.convert('L')
        
        # Resize to 28x28
        img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized)
        
        # Invert colors (white background to black, black digit to white)
        img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict_digit(self):
        """Predict the drawn digit"""
        if self.model is None:
            self.result_var.set("Model not loaded!")
            return
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image()
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            
            # Update result
            self.result_var.set(f"Predicted Digit: {predicted_digit} "
                               f"(Confidence: {confidence:.1f}%)")
            
            # Show confidence scores for all digits
            self.show_confidence_scores(prediction[0])
            
        except Exception as e:
            self.result_var.set(f"Prediction error: {str(e)}")
    
    def show_confidence_scores(self, predictions):
        """Display confidence scores for all digits"""
        # Clear previous scores
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        
        # Title for confidence scores
        conf_title = tk.Label(self.confidence_frame, text="Confidence Scores:", 
                             font=("Arial", 10, "bold"))
        conf_title.pack()
        
        # Create frame for scores
        scores_frame = tk.Frame(self.confidence_frame)
        scores_frame.pack()
        
        # Display scores for each digit
        for digit in range(10):
            confidence = predictions[digit] * 100
            color = "green" if digit == np.argmax(predictions) else "black"
            
            score_label = tk.Label(scores_frame, 
                                  text=f"{digit}: {confidence:.1f}%",
                                  font=("Arial", 9), fg=color)
            score_label.grid(row=digit//5, column=digit%5, padx=5, pady=2)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function to run the drawing app"""
    print("Starting Digit Drawing Application...")
    app = DigitDrawingApp()
    app.run()

if __name__ == "__main__":
    main()