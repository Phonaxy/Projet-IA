from tensorflow import keras
from keras import layers
from keras.datasets import mnist

# Charger et préparer les données
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation

# Définir le modèle
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()  # Affiche l’architecture

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner le modèle
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1,
    batch_size=32
)

# Évaluer sur le test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
