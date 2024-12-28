import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import shutil
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage

COLORS = {
    "primary": "#FF69B4",
    "secondary": "#FF91C7",
    "accent": "#FF1493",
    "background": "#FFF0F5",
    "text": "#4A4A4A",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C"
}

model_path = "cnn_breast_cancer_model.keras"
cnn_model = load_model(model_path)

training_history = {
    "accuracy": [0.7, 0.8, 0.85, 0.9],
    "val_accuracy": [0.68, 0.75, 0.82, 0.87],
    "loss": [0.6, 0.5, 0.4, 0.35],
    "val_loss": [0.65, 0.55, 0.45, 0.38],
}

label_map = {0: "Non-Cancer", 1: "Early Phase", 2: "Middle Phase"}

def create_ribbon_canvas(master):
    canvas = tk.Canvas(master, width=100, height=100, bg=COLORS["background"], highlightthickness=0)
    canvas.create_arc(20, 20, 80, 80, start=45, extent=180, fill=COLORS["primary"])
    canvas.create_arc(20, 20, 80, 80, start=225, extent=180, fill=COLORS["primary"])
    return canvas

class CustomButton(tk.Button):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            relief=tk.FLAT,
            borderwidth=0,
            padx=25,
            pady=12,
            font=("Helvetica", 12, "bold"),
            cursor="hand2"
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(bg=COLORS["secondary"])

    def on_leave(self, event):
        self.configure(bg=COLORS["primary"])

def predict_image(image_path):
    image = load_img(image_path)
    image_array = img_to_array(image)
    
    resized_image = cv2.resize(image_array, (128, 128))
    gray_image = cv2.cvtColor(resized_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_image = gray_image / 255.0
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = np.expand_dims(gray_image, axis=0)
    
    predictions = cnn_model.predict(gray_image)
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label] * 100
    return label_map[predicted_label], confidence, predictions[0]

class ModernFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=COLORS["background"],
            relief=tk.FLAT,
            borderwidth=0,
            padx=20,
            pady=20
        )

class BreastCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Detection System")
        self.root.geometry("1200x900")
        self.root.configure(bg=COLORS["background"])
        
        self.main_container = ModernFrame(self.root)
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)

        self.art_frame = tk.Frame(self.main_container, bg=COLORS["background"])
        self.art_frame.pack(pady=10)
        
        self.ribbon = create_ribbon_canvas(self.art_frame)
        self.ribbon.pack(side=tk.LEFT, padx=10)

        self.header = tk.Label(
            self.art_frame,
            text="Breast Cancer Detection System\nEarly Detection Saves Lives",
            font=("Helvetica", 24, "bold"),
            fg=COLORS["primary"],
            bg=COLORS["background"]
        )
        self.header.pack(side=tk.LEFT, padx=20)

        self.control_panel = ModernFrame(self.main_container)
        self.control_panel.pack(fill="x", pady=15)

        self.buttons_frame = tk.Frame(self.control_panel, bg=COLORS["background"])
        self.buttons_frame.pack(pady=10)

        self.current_analysis = None
        self.current_image_path = None

        buttons = [
            ("Upload Image", self.upload_image),
            ("Training History", self.show_training_history_popup),
            ("Download Report", self.download_report)
        ]
        
        for text, command in buttons:
            btn = CustomButton(
                self.buttons_frame,
                text=text,
                command=command,
                bg=COLORS["primary"],
                fg="white"
            )
            btn.pack(side=tk.LEFT, padx=15)
            if text == "Download Report":
                self.download_button = btn
                self.download_button.config(state="disabled")

        self.content_frame = tk.Frame(self.main_container, bg=COLORS["background"])
        self.content_frame.pack(expand=True, fill="both", pady=20)

        self.left_frame = ModernFrame(self.content_frame)
        self.left_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=(0, 10))

        self.right_frame = ModernFrame(self.content_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.image_display = tk.Frame(self.right_frame, width=500, height=500, bg="#FFF0F5")
        self.image_display.pack_propagate(False)
        self.image_display.pack(pady=10)

        self.prediction_label = tk.Label(
            self.left_frame,
            text="Please upload the image for analysis",
            font=("Helvetica", 16),
            fg=COLORS["text"],
            bg=COLORS["background"]
        )
        self.prediction_label.pack(pady=15)

        self.probabilities_text = tk.Text(
            self.left_frame,
            height=8,
            width=40,
            font=("Helvetica", 12),
            bg="#FFF0F5",
            relief=tk.FLAT
        )
        self.probabilities_text.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Mammogram Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.current_image_path = file_path
        phase, confidence, predictions = predict_image(file_path)

        self.current_analysis = {
            "phase": phase,
            "confidence": confidence,
            "predictions": predictions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": file_path
        }

        self.download_button.config(state="normal")

        result_colors = {
            "Non-Cancer": COLORS["success"],
            "Early Phase": COLORS["warning"],
            "Middle Phase": COLORS["danger"]
        }
        result_color = result_colors.get(phase, COLORS["text"])

        for widget in self.left_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.destroy()

        diagnosis_frame = tk.Frame(
            self.left_frame,
            bg=result_color,
            padx=25,
            pady=15
        )
        diagnosis_frame.pack(fill="x", pady=15)

        labels = [
            (f"Diagnosis: {phase}", ("Helvetica", 18, "bold")),
            (f"Confidence: {confidence:.2f}%", ("Helvetica", 14))
        ]
        
        for text, font in labels:
            tk.Label(
                diagnosis_frame,
                text=text,
                fg="white",
                bg=result_color,
                font=font
            ).pack()

        self.update_probabilities(predictions)
        self.display_image(file_path)

        view_button = CustomButton(
            self.right_frame,
            text="View Full Image",
            command=lambda: self.display_image_popup(file_path, phase, confidence),
            bg=COLORS["primary"],
            fg="white"
        )
        view_button.pack(pady=5)

    def display_image(self, file_path):
        for widget in self.image_display.winfo_children():
            widget.destroy()

        image = load_img(file_path)
        image_array = img_to_array(image).astype(np.uint8)
        resized_image = cv2.resize(image_array, (500, 500))

        fig = Figure(figsize=(5, 5), facecolor="#FFF0F5")
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.image_display)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def display_image_popup(self, file_path, phase, confidence):
        popup = tk.Toplevel(self.root)
        popup.title("Mammogram Image")
        popup.geometry("800x800")
        
        image = load_img(file_path)
        image_array = img_to_array(image).astype(np.uint8)
        resized_image = cv2.resize(image_array, (800, 800))

        fig = Figure(figsize=(8, 8), facecolor="#FFF0F5")
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")
        
        label_text = f"{phase} (Confidence: {confidence:.2f}%)"
        label = tk.Label(popup, text=label_text, font=("Helvetica", 18, "bold"), fg=COLORS["primary"])
        label.pack(pady=10)

    def update_probabilities(self, predictions):
        self.probabilities_text.delete(1.0, tk.END)

        for i, (phase, prob) in enumerate(zip(label_map.values(), predictions)):
            self.probabilities_text.insert(tk.END, f"{phase}: {prob * 100:.2f}%\n")

    def show_training_history_popup(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Model Training History")
        history_window.geometry("600x400")

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        ax.plot(training_history["accuracy"], label="Training Accuracy", color=COLORS["success"])
        ax.plot(training_history["val_accuracy"], label="Validation Accuracy", color=COLORS["warning"])
        ax.plot(training_history["loss"], label="Training Loss", color=COLORS["danger"])
        ax.plot(training_history["val_loss"], label="Validation Loss", color=COLORS["primary"])

        ax.set_title("Model Training History")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy / Loss")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=history_window)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both", pady=20)

    def download_report(self):
        if not self.current_analysis:
            return
        timestamp = self.current_analysis["timestamp"].replace(":", "-")
        report_filename = f"breast_cancer_report_{timestamp}.pdf"
        self.generate_pdf_report(report_filename)

    def generate_pdf_report(self, filename):
        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        
        c.drawString(30, 770, "Breast Cancer Detection Report")
        c.setFont("Helvetica", 12)
        
        c.drawString(30, 740, f"Date: {self.current_analysis['timestamp']}")
        c.drawString(30, 710, f"Diagnosis: {self.current_analysis['phase']}")
        c.drawString(30, 680, f"Confidence: {self.current_analysis['confidence']:.2f}%")
        
        c.drawString(30, 650, "Model Prediction Probabilities:")
        y_position = 620
        
        for i, (phase, prob) in enumerate(zip(label_map.values(), self.current_analysis['predictions'])):
            c.drawString(30, y_position, f"{phase}: {prob * 100:.2f}%")
            y_position -= 20
        
        c.drawImage(self.current_analysis['image_path'], 30, 350, width=150, height=150)
        
        c.save()

        messagebox.showinfo("Report Downloaded", f"Report saved as {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()
