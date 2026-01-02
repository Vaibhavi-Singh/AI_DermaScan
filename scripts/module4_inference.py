import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ---------------- #
MODEL_PATH = Path("models/efficientnet_b0.keras")
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["clear_skin", "dark_spots", "puffy_eyes", "wrinkles"]
CONFIDENCE_THRESHOLD = 0.60
# ---------------------------------------- #

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

    print("📌 Loading trained EfficientNet-B0 model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    return model

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=IMAGE_SIZE)
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img, img_array
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


# def predict_image(model, img_path):
#     img, img_tensor = preprocess_image(img_path)
#     preds = model.predict(img_tensor, verbose=0)[0]

#     print("\n📊 Class-wise Probabilities:")
#     for cls, prob in zip(CLASS_NAMES, preds):
#         print(f"  {cls:<12}: {prob * 100:.2f}%")

#     top_idx = np.argmax(preds)
#     top_class = CLASS_NAMES[top_idx]
#     top_conf = float(preds[top_idx])

#     print("\n🔍 Final Decision:")
#     if top_conf < CONFIDENCE_THRESHOLD:
#         decision_text = "Low confidence – model unsure"
#         print(f"👉 Predicted Class : {top_class}")
#         print(f"👉 Confidence     : {top_conf * 100:.2f}%")
#         print("⚠️ Model is unsure. Please provide a clearer image.")
#     else:
#         decision_text = "High confidence"
#         print(f"👉 Predicted Class : {top_class}")
#         print(f"👉 Confidence     : {top_conf * 100:.2f}%")

#     # ---------- SHOW IMAGE ----------
#     plt.figure(figsize=(5, 5))
#     plt.imshow(img)
#     plt.title(f"{top_class} ({top_conf:.2f})\n{decision_text}")
#     plt.axis("off")
#     plt.show()

def predict_image(model, img_path):
    img, img_tensor = preprocess_image(img_path)
    preds = model.predict(img_tensor, verbose=0)[0]

    print("\n📊 Class-wise Probabilities:")
    for cls, prob in zip(CLASS_NAMES, preds):
        print(f"  {cls:<12}: {prob * 100:.2f}%")

    top_idx = np.argmax(preds)
    top_class = CLASS_NAMES[top_idx]
    top_conf = preds[top_idx] * 100  # percentage

    print("\n🔍 Final Decision:")
    print(f"👉 Predicted Class : {top_class}")
    print(f"👉 Confidence     : {top_conf:.2f}%")

    if top_conf < CONFIDENCE_THRESHOLD * 100:
        status_text = "Low confidence prediction"
    else:
        status_text = "High confidence prediction"

    # =========================
    # IMAGE OVERLAY
    # =========================
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")

    plt.title(
        f"{top_class.upper()} ({top_conf:.2f}%)\n{status_text}",
        fontsize=12,
        color="red" if top_conf < 60 else "green"
    )

    plt.show()


def main():
    model = load_model()
    img_path = input("\nEnter image path to test: ").strip()

    if not Path(img_path).exists():
        print("❌ Image path not found!")
        return

    predict_image(model, img_path)

if __name__ == "__main__":
    main()
