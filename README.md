# image_sharpening

# 🖼️ Image Sharpening Web App using Knowledge Distillation

This project is a full-stack web application that sharpens blurry images using a deep learning model trained with **Knowledge Distillation**. The frontend is built in HTML/JS, while the backend uses Flask and PyTorch.

---

## 🔧 Features

- 📁 Upload `.png` or `.jpeg` images
- 🧠 Applies a trained lightweight StudentNet model
- 📈 Displays **SSIM (Structural Similarity)** score
- 🖼️ Shows original and sharpened image side-by-side
- 🚫 Supports Live Server without auto-refresh issues

---

## 📁 Project Structure

🔹 Frontend (Live Server or Python)
Option 1: With Live Server extension in VS Code
Option 2: Python static server:

bash
Copy
Edit
cd frontend
python3 -m http.server 5500
Then visit:

cpp
Copy
Edit
http://127.0.0.1:5500/

🧠 Training the Model
To retrain the student model using distillation:

bash
Copy
Edit
cd backend
python train.py
Ensure your dataset is structured like:
dataset/lr/ → Low-resolution or blurry images
dataset/hr/ → High-resolution (ground truth) images

🛠️ Dependencies
Install requirements with:

bash
Copy
Edit
pip install -r requirements.txt
Major libraries:

flask

flask-cors

torch

opencv-python

scikit-image

tqdm

