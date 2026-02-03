from tensorflow import keras
from keras import layers
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================================
# CHARGER DATASET CUSTOM (remplace mnist.load_data())
# ============================================================================

def load_custom_dataset(data_dir="/home/docker/Work/data/custom_digits"):
    """Charge le dataset custom depuis les dossiers 0-9"""
    data_dir = Path(data_dir)
    
    images = []
    labels = []
    
    # Charger chaque chiffre (0-9)
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        
        # Charger toutes les images BMP
        for img_path in sorted(digit_dir.glob("*.bmp")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                labels.append(digit)
    
    X = np.array(images, dtype='float32')
    y = np.array(labels)
    
    # Split train/test (80/20)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (x_train, y_train), (x_test, y_test)

# ============================================================================
# RESTE DU CODE IDENTIQUE
# ============================================================================

# Charger et préparer les données (CHANGEMENT ICI)
(x_train, y_train), (x_test, y_test) = load_custom_dataset()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation

print(f"✅ Dataset chargé: {len(x_train)} train, {len(x_test)} test")

# Définir le modèle
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes
])
model.summary()  # Affiche l'architecture

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner le modèle
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.1,
    batch_size=8
)

# Évaluer sur le test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")