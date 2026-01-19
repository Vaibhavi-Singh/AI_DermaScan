# 🧬 DermalScan: AI Facial Skin Detection App

## 📌 Project Overview
DermalScan is an AI-powered facial skin analysis system that detects and classifies skin conditions using deep learning and computer vision techniques.  
The application supports both **image upload** and **webcam capture** for real-time skin analysis.

## 🎯 Skin Conditions Detected
- Wrinkles  
- Dark Spots  
- Puffy Eyes  
- Clear Skin  

## ⚙️ Technology Stack
- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **Deep Learning:** TensorFlow, Keras (EfficientNet-B0)  
- **Computer Vision:** OpenCV  
- **Model Type:** CNN (Transfer Learning)  
- **Deployment:** Local Flask Web App  

## 🧠 Model Architecture
- EfficientNet-B0 pretrained on ImageNet  
- Fine-tuned on facial skin dataset  
- Softmax classification (4 classes)  
- Confidence thresholding to avoid fake predictions  

## 🖼 Features
- Upload facial image for analysis  
- Real-time webcam capture  
- Annotated output image with prediction & confidence  
- CSV report generation  
- Download annotated results  
- Invalid image detection (non-face / animals rejected)  

## 📊 Output
- Predicted skin condition  
- Confidence score  
- Class-wise probabilities  
- Annotated image  
- CSV report log  

## 🚀 How to Run
```bash
pip install -r requirements.txt
python app.py


Open browser:
http://127.0.0.1:5000

📁 Project Structure
DermalScan/
│── app.py
│── models/
│── static/
│   ├── uploads/
│   ├── annotated/
│   ├── reports/
│── templates/
│   └── index.html
│── README.md



