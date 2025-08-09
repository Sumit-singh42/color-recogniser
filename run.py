from PIL import Image
import numpy as np
import tensorflow as tf

# Load model & labels
model = tf.keras.models.load_model("color_cnn.h5")
labels = np.load("color_labels.npy")
def predict_color(image_path):
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred_idx = np.argmax(model.predict(arr))
    return labels[pred_idx]
print(predict_color("color_test_1.png"))