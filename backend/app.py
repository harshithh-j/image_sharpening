from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import cv2
import torch
import numpy as np
from model.student_model import StudentNet
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
CORS(app)

# Save outputs here
UPLOAD_FOLDER = '../outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = StudentNet()
model.load_state_dict(torch.load("student_weights.pth", map_location='cpu'))
model.eval()

@app.route('/sharpen', methods=['POST'])
def sharpen():
    file = request.files['image']
    filename = file.filename.lower()

    # ✅ Validate extension
    if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
        return jsonify({"error": "Only PNG and JPEG files are supported."}), 400

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Load image and normalize
    img = cv2.imread(input_path).astype(np.float32) / 255.0

    # Convert to tensor
    input_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)

    # Model inference
    with torch.no_grad():
        output = model(input_tensor).squeeze().permute(1, 2, 0).numpy()

    # Save output image
    output_filename = 'sharpened_' + filename
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, (output * 255).astype(np.uint8))

    # Compute SSIM
    score = ssim(img, output, channel_axis=2, data_range=1.0)

    return jsonify({
        'output_url': f'static/outputs/{output_filename}',
        'ssim': float(score)
    })

@app.route('/preview/<filename>')
def preview(filename):
    return send_file(os.path.join('../outputs', filename), mimetype='image/jpeg')


@app.route('/static/outputs/<path:filename>')
def download_file(filename):
    return send_from_directory('../outputs', filename)

if __name__ == '__main__':
    app.run(debug=True)
