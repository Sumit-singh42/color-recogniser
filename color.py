# pip install tensorflow numpy pandas pillow sklearn
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# some color names from github
url = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
df = pd.read_csv(url)

IMG_SIZE = 32
SAMPLES_PER_COLOR = 120 

X, y = [], []
for _, row in df.iterrows():
    r, g, b, name = int(row[3]), int(row[4]), int(row[5]), row[1]
    for _ in range(SAMPLES_PER_COLOR):
        r_noise = np.clip(r + np.random.randint(-5, 5), 0, 255)
        g_noise = np.clip(g + np.random.randint(-5, 5), 0, 255)
        b_noise = np.clip(b + np.random.randint(-5, 5), 0, 255)
        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img[:, :, 0] = r_noise
        img[:, :, 1] = g_noise
        img[:, :, 2] = b_noise
        X.append(img)
        y.append(name)

X = np.array(X) / 255.0 
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test_accuracy: {acc*100:.2f}%")

model.save("color_cnn.h5")
np.save("color_labels.npy", le.classes_)