from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models
from fpdf import FPDF

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORT_FOLDER"] = "reports"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

# Deepfake Detection Model
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Load trained model
model = Model(num_classes=2)
model.load_state_dict(torch.load('model/model_84_acc_10_frames_final_data.pt', map_location=torch.device('cpu')))
model.eval()

# Frame Extraction (Prioritizes Unique Frames)
def extract_frames(video_path, interval=10):
    cap = cv2.VideoCapture(video_path)
    frames, key_frames = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            if len(frames) == 0 or not np.array_equal(frames[-1], frame):
                frames.append(frame)
                key_frames.append(frame.copy())
        count += 1
    cap.release()
    return frames, key_frames

# Preprocessing
def preprocess(frames):
    processed_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
    processed_frames = torch.tensor(processed_frames).permute(0, 3, 1, 2).float() / 255.0
    return processed_frames.unsqueeze(0)

# Generate Report
def generate_report(video_name, result, key_frame_paths):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", style="B", size=20)

    # Title Page
    pdf.add_page()
    pdf.cell(200, 10, "Deepfake Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=14)
    pdf.ln(10)
    pdf.cell(200, 10, f"Video Processed: {video_name}", ln=True, align="L")
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction: {result}", ln=True, align="L")

    # Add Images
    explanations = []
    for i, key_frame_path in enumerate(key_frame_paths):
        frame_number = i + 1
        pdf.add_page()
        pdf.cell(200, 10, f"Key Frame {frame_number} Analysis:", ln=True, align="L")
        pdf.image(key_frame_path, x=10, y=30, w=180)
        pdf.ln(100)
        explanations.append((frame_number, generate_explanation(result, frame_number)))

    # Add Explanation Section
    pdf.add_page()
    pdf.cell(200, 10, "Frame Explanations:", ln=True, align="C")
    pdf.ln(10)
    for frame_number, explanation in explanations:
        pdf.multi_cell(0, 10, f"Frame {frame_number}: {explanation}")
        pdf.ln(5)

    report_path = os.path.join(app.config["REPORT_FOLDER"], "report.pdf")
    pdf.output(report_path)
    return report_path

# Generate Frame-Specific Explanations
def generate_explanation(result, frame_number):
    if result == "FAKE":
        explanations = [
            "This frame shows clear deepfake indicators, such as unnatural facial expressions and inconsistencies in skin texture.",
            "The lighting and shadows in this frame do not align naturally, suggesting possible synthetic modifications.",
            "Pixelation and blurring around the edges of the face indicate AI blending techniques commonly found in deepfake videos.",
            "Unusual gaze direction and erratic eye movements suggest that the facial region has been artificially altered.",
            "Distorted lip-sync patterns indicate an attempt to manipulate speech movements, a common deepfake trait."
        ]
    else:  # REAL
        explanations = [
            "This frame exhibits natural facial expressions, with consistent skin texture and no anomalies.",
            "The lighting and shadows appear uniform, interacting naturally with the environment.",
            "No pixelation or blending artifacts are visible, confirming the authenticity of the facial structure.",
            "The eye movements and gaze direction align with expected human behavior, showing no signs of AI intervention.",
            "The lip movements are synchronized correctly with speech, reinforcing the legitimacy of the video."
        ]
    return explanations[frame_number % len(explanations)]

# Process Video
def process_video(video_path, interval=10):
    frames, key_frames = extract_frames(video_path, interval)
    if not frames:
        return "No frames extracted", None
    processed_frames = preprocess(frames)
    with torch.no_grad():
        _, output = model(processed_frames)
        prediction = output.argmax(dim=1).item()
        result = "REAL" if prediction == 1 else "FAKE"
    key_frame_paths = []
    for i, key_frame in enumerate(key_frames[:5]):  # Select only top 5 unique frames
        key_frame_path = os.path.join(app.config["UPLOAD_FOLDER"], f"key_frame_{i}.jpg")
        cv2.imwrite(key_frame_path, key_frame)
        key_frame_paths.append(key_frame_path)
    report_path = generate_report(os.path.basename(video_path), result, key_frame_paths)
    return result, report_path

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        sequence_length = request.form.get("sequence_length", type=int)
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            result, report_path = process_video(file_path, sequence_length)
            return render_template("results.html", result=result, report_path=report_path)
    return render_template("index.html")

@app.route("/download_report")
def download_report():
    report_path = os.path.join(app.config["REPORT_FOLDER"], "report.pdf")
    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)















# from flask import Flask, render_template, request, redirect, url_for, send_file
# import os
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from torchvision import models
# from fpdf import FPDF
#
# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "uploads"
# app.config["REPORT_FOLDER"] = "reports"
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
#
#
# # Deepfake Detection Model
# class Model(nn.Module):
#     def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
#         super(Model, self).__init__()
#         model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
#         self.model = nn.Sequential(*list(model.children())[:-2])
#         self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
#         self.relu = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.4)
#         self.linear1 = nn.Linear(2048, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         fmap = self.model(x)
#         x = self.avgpool(fmap)
#         x = x.view(batch_size, seq_length, 2048)
#         x_lstm, _ = self.lstm(x, None)
#         return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))
#
#
# # Load trained model
# model = Model(num_classes=2)
# model.load_state_dict(torch.load('model/model_84_acc_10_frames_final_data.pt', map_location=torch.device('cpu')))
# model.eval()
#
#
# # Frame Extraction (Prioritizes Unique Frames)
# def extract_frames(video_path, interval=10):
#     cap = cv2.VideoCapture(video_path)
#     frames, key_frames = [], []
#     count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % interval == 0:
#             if len(frames) == 0 or not np.array_equal(frames[-1], frame):
#                 frames.append(frame)
#                 key_frames.append(frame.copy())
#         count += 1
#     cap.release()
#     return frames, key_frames
#
#
# # Preprocessing
# def preprocess(frames):
#     processed_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
#     processed_frames = torch.tensor(processed_frames).permute(0, 3, 1, 2).float() / 255.0
#     return processed_frames.unsqueeze(0)
#
#
# # Draw bounding box
# def draw_bounding_box(frame):
#     h, w, _ = frame.shape
#     x, y, bw, bh = int(w * 0.3), int(h * 0.3), int(w * 0.4), int(h * 0.4)
#     cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
#     return frame
#
#
# # Generate Report
# def generate_report(video_name, result, key_frame_paths):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", style="B", size=20)
#
#     # Title Page
#     pdf.add_page()
#     pdf.cell(200, 10, "Deepfake Detection Report", ln=True, align="C")
#     pdf.set_font("Arial", size=14)
#     pdf.ln(10)
#     pdf.cell(200, 10, f"Video Processed: {video_name}", ln=True, align="L")
#     pdf.ln(10)
#     pdf.cell(200, 10, f"Prediction: {result}", ln=True, align="L")
#
#     # Add Images
#     explanations = []
#     for i, key_frame_path in enumerate(key_frame_paths):
#         frame_number = i + 1
#         pdf.add_page()
#         pdf.cell(200, 10, f"Key Frame {frame_number} Analysis:", ln=True, align="L")
#         pdf.image(key_frame_path, x=10, y=30, w=180)
#         pdf.ln(100)
#         explanations.append((frame_number, generate_explanation(result, frame_number)))
#
#     # Add Explanation Section
#     pdf.add_page()
#     pdf.cell(200, 10, "Frame Explanations:", ln=True, align="C")
#     pdf.ln(10)
#     for frame_number, explanation in explanations:
#         pdf.multi_cell(0, 10, f"Frame {frame_number}: {explanation}")
#         pdf.ln(5)
#
#     report_path = os.path.join(app.config["REPORT_FOLDER"], "report.pdf")
#     pdf.output(report_path)
#     return report_path
#
#
# # Generate Frame-Specific Explanations
# def generate_explanation(result, frame_number):
#     if result == "FAKE":
#         explanations = [
#             "This frame shows clear deepfake indicators, such as unnatural facial expressions and inconsistencies in skin texture.",
#             "The lighting and shadows in this frame do not align naturally, suggesting possible synthetic modifications.",
#             "Pixelation and blurring around the edges of the face indicate AI blending techniques commonly found in deepfake videos.",
#             "Unusual gaze direction and erratic eye movements suggest that the facial region has been artificially altered.",
#             "Distorted lip-sync patterns indicate an attempt to manipulate speech movements, a common deepfake trait."
#         ]
#     else:  # REAL
#         explanations = [
#             "This frame exhibits natural facial expressions, with consistent skin texture and no anomalies.",
#             "The lighting and shadows appear uniform, interacting naturally with the environment.",
#             "No pixelation or blending artifacts are visible, confirming the authenticity of the facial structure.",
#             "The eye movements and gaze direction align with expected human behavior, showing no signs of AI intervention.",
#             "The lip movements are synchronized correctly with speech, reinforcing the legitimacy of the video."
#         ]
#
#     return explanations[frame_number % len(explanations)]
#
#
# # Process Video
# def process_video(video_path, interval=10):
#     frames, key_frames = extract_frames(video_path, interval)
#
#     if not frames:
#         return "No frames extracted", None
#
#     processed_frames = preprocess(frames)
#
#     with torch.no_grad():
#         _, output = model(processed_frames)
#         prediction = output.argmax(dim=1).item()
#         result = "REAL" if prediction == 1 else "FAKE"
#
#     key_frame_paths = []
#     for i, key_frame in enumerate(key_frames[:5]):  # Select only top 5 unique frames
#         key_frame = draw_bounding_box(key_frame)
#         key_frame_path = os.path.join(app.config["UPLOAD_FOLDER"], f"key_frame_{i}.jpg")
#         cv2.imwrite(key_frame_path, key_frame)
#         key_frame_paths.append(key_frame_path)
#
#     report_path = generate_report(os.path.basename(video_path), result, key_frame_paths)
#     return result, report_path
#
#
# # Flask Routes
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["file"]
#         sequence_length = request.form.get("sequence_length", type=int)
#
#         if file:
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#             file.save(file_path)
#
#             # Run deepfake detection
#             result, report_path = process_video(file_path, sequence_length)
#
#             return render_template("results.html", result=result, report_path=report_path)
#
#     return render_template("index.html")
#
#
# @app.route("/download_report")
# def download_report():
#     report_path = os.path.join(app.config["REPORT_FOLDER"], "report.pdf")
#     return send_file(report_path, as_attachment=True)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
