# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from pathlib import Path
# import os
# from PIL import Image, ImageDraw, ImageFont
# import csv
# from datetime import datetime


# import cv2



# app = Flask(__name__)


# # ---------------- CONFIG ----------------
# MODEL_PATH = Path("models/efficientnet_b0.keras")
# UPLOAD_FOLDER = "static/uploads"
# IMAGE_SIZE = (224, 224)

# CLASS_NAMES = ["clear_skin", "dark_spots", "puffy_eyes", "wrinkles"]
# CONFIDENCE_THRESHOLD = 0.60
# CSV_REPORT_PATH = "static/reports/predictions_report.csv"
# os.makedirs("static/reports", exist_ok=True)

# # ----------------------------------------

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# print("📌 Loading trained model...")
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# print("✅ Model loaded successfully!")


# # def predict_image(img_path):
# #     img = image.load_img(img_path, target_size=IMAGE_SIZE)
# #     img = image.img_to_array(img) / 255.0
# #     img = np.expand_dims(img, axis=0)

# #     preds = model.predict(img)[0]
# #     class_index = np.argmax(preds)
# #     confidence = float(preds[class_index])

# #     return CLASS_NAMES[class_index], confidence, preds


# # def predict_and_annotate(img_path):
# #     # Load image for model
# #     img = image.load_img(img_path, target_size=IMAGE_SIZE)
# #     img_array = image.img_to_array(img) / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)

# #     preds = model.predict(img_array)[0]
# #     class_index = np.argmax(preds)
# #     confidence = float(preds[class_index])

# #     predicted_class = CLASS_NAMES[class_index]

# #     # ---------------------------
# #     # ANNOTATION PART (PIL)
# #     # ---------------------------
# #     original_img = Image.open(img_path).convert("RGB")
# #     draw = ImageDraw.Draw(original_img)

# #     text = f"{predicted_class.upper()}  ({confidence*100:.2f}%)"

# #     color = "red" if confidence < CONFIDENCE_THRESHOLD else "green"

# #     try:
# #         font = ImageFont.truetype("arial.ttf", 28)
# #     except:
# #         font = ImageFont.load_default()

# #     draw.rectangle(
# #         [(10, 10), (10 + 420, 60)],
# #         fill="black"
# #     )
# #     draw.text((20, 20), text, fill=color, font=font)

# #     # Save annotated image
# #     annotated_path = img_path.replace(".jpg", "_annotated.jpg").replace(".png", "_annotated.png")
# #     original_img.save(annotated_path)

# #     return predicted_class, confidence, preds, annotated_path

# def predict_and_annotate(img_path):
#     img = image.load_img(img_path, target_size=IMAGE_SIZE)
#     img_arr = image.img_to_array(img) / 255.0
#     img_arr = np.expand_dims(img_arr, axis=0)

#     preds = model.predict(img_arr, verbose=0)[0]
#     class_index = np.argmax(preds)
#     confidence = float(preds[class_index])
#     label = CLASS_NAMES[class_index]

#     # =========================
#     # ANNOTATION
#     # =========================
#     original = cv2.imread(img_path)
#     original = cv2.resize(original, (224, 224))

#     text = f"{label.upper()} ({confidence*100:.2f}%)"
#     color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

#     cv2.putText(
#         original,
#         text,
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         color,
#         2
#     )

#     # Save annotated image
#     annotated_name = f"annotated_{Path(img_path).name}"
#     annotated_path = Path("static/annotated") / annotated_name
#     cv2.imwrite(str(annotated_path), original)

#     return label, confidence, preds, str(annotated_path)




# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     result = None
# #     confidence = None
# #     probs = None
# #     img_path = None
# #     annotated_img = None   # ✅ ADD THIS LINE
# #     warning = None


# #     if request.method == "POST":
# #         file = request.files.get("image")

# #         if file:
# #             img_path = os.path.join(UPLOAD_FOLDER, file.filename)
# #             file.save(img_path)

# #             # result, confidence, probs = predict_image(img_path)
# #             result, confidence, probs, annotated_img = predict_and_annotate(img_path)


# #             if confidence < CONFIDENCE_THRESHOLD:
# #                 warning = "⚠️ Low confidence prediction. Model is unsure."

# #     # return render_template(
# #     #     "index.html",
# #     #     result=result,
# #     #     confidence=confidence,
# #     #     probs=probs,
# #     #     img_path=img_path,
# #     #     class_names=CLASS_NAMES,
# #     #     warning=warning
# #     # )
# #     return render_template(
# #     "index.html",
# #     result=result,
# #     confidence=confidence,
# #     probs=probs,
# #     img_path=img_path,
# #     annotated_img=annotated_img,
# #     class_names=CLASS_NAMES,
# #     warning=warning
# # )
# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     confidence = None
#     probs = None
#     img_path = None
#     annotated_img = None
#     warning = None

#     if request.method == "POST":
#         file = request.files.get("image")

#         if file:
#             img_path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(img_path)

#             result, confidence, probs, annotated_img = predict_and_annotate(img_path)

#             if confidence < CONFIDENCE_THRESHOLD:
#                 warning = "⚠️ Low confidence prediction. Model is unsure."

#     return render_template(
#         "index.html",
#         result=result,
#         confidence=confidence,
#         probs=probs,
#         img_path=img_path,
#         annotated_img=annotated_img,
#         class_names=CLASS_NAMES,
#         warning=warning
#     )



# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path
import os
import cv2
from datetime import datetime

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = Path("models/efficientnet_b0.keras")
UPLOAD_FOLDER = "static/uploads"
ANNOTATED_FOLDER = "static/annotated"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["clear_skin", "dark_spots", "puffy_eyes", "wrinkles"]
CONFIDENCE_THRESHOLD = 0.60
# ----------------------------------------

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Load model
print("📌 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# Load face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- VALIDATION ----------------
def is_valid_face_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


# ---------------- PREDICTION + ANNOTATION ----------------
def predict_and_annotate(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)[0]
    class_index = np.argmax(preds)
    confidence = float(preds[class_index])
    label = CLASS_NAMES[class_index]

    # OpenCV annotation
    original = cv2.imread(img_path)
    original = cv2.resize(original, IMAGE_SIZE)

    text = f"{label.upper()} ({confidence*100:.2f}%)"
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

    cv2.putText(
        original,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    annotated_name = f"annotated_{Path(img_path).name}"
    annotated_path = Path(ANNOTATED_FOLDER) / annotated_name
    cv2.imwrite(str(annotated_path), original)

    return label, confidence, preds, str(annotated_path)


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    probs = None
    img_path = None
    annotated_img = None
    warning = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            if not is_valid_face_image(img_path):
                warning = "❌ Invalid image. Please upload a clear human face image."
            else:
                result, confidence, probs, annotated_img = predict_and_annotate(img_path)

                if confidence < CONFIDENCE_THRESHOLD:
                    warning = "⚠️ Low confidence prediction. Model is unsure."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        probs=probs,
        img_path=img_path,
        annotated_img=annotated_img,
        class_names=CLASS_NAMES,
        warning=warning
    )


@app.route("/webcam")
def webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "❌ Webcam not accessible"

    img_name = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, img_name)
    cv2.imwrite(img_path, frame)

    if not is_valid_face_image(img_path):
        return render_template(
            "index.html",
            warning="❌ Invalid webcam image. Please align your face properly."
        )

    result, confidence, probs, annotated_img = predict_and_annotate(img_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        probs=probs,
        img_path=img_path,
        annotated_img=annotated_img,
        class_names=CLASS_NAMES
    )


if __name__ == "__main__":
    app.run(debug=True)

