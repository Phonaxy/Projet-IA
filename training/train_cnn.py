from tensorflow import keras
from keras import layers
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

print("CNN TEST 2 : 3 Conv (16->32->32) + Dense64 + Dropout 0.5 + L2")

def load_custom_dataset(data_dir="/home/docker/Work/data/custom_digits_inverted"):
    data_dir = Path(data_dir)
    images = []
    labels = []
    
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        for img_path in sorted(digit_dir.glob("*.bmp")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(digit)
    
    X = np.array(images, dtype='float32')
    y = np.array(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_custom_dataset()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

print(f"Dataset: {len(x_train)} train, {len(x_test)} test")

# === CNN 3 BLOCS CONV + CLASSIFICATEUR LÉGER ===
model = keras.Sequential([
    # Bloc Conv 1 : 28×28×1 → 14×14×16
    layers.Conv2D(16, 3, padding='same', activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001),
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    
    # Bloc Conv 2 : 14×14×16 → 7×7×32
    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D(2),
    
    # Bloc Conv 3 : 7×7×32 → 7×7×32 (pas de pooling, juste profondeur)
    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001)),
    
    # Classificateur léger
    layers.Flatten(),                    # 7×7×32 = 1568
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("\n Archi: Conv16->Pool->Conv32->Pool->Conv32->Flat->Dense64->Drop0.5->10")
print("3 couches Conv (au lieu de 2)")
print("Dense: 64 (au lieu de 128)")
print("Dropout: 0.5")
print("L2: 0.001 sur TOUTES les couches (Conv + Dense)")
print("Adam lr: 0.0003")
print("Batch size: 16")

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.0003)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
]

history = model.fit(
    x_train, y_train,
    epochs=300,
    validation_data=(x_test, y_test),
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)

# === RÉSULTATS ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

stopped_epoch = len(history.history['loss'])
best_val_acc = max(history.history['val_accuracy']) * 100
best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
final_train_acc = history.history['accuracy'][-1] * 100

print("Résults du CNN test 2")
print("="*70)
print(f"test accuracy:       {test_acc*100:.1f}%")
print(f"Meilleure val_acc:   {best_val_acc:.1f}% (epoch {best_val_acc_epoch})")
print(f"Epoch d'arrêt:       {stopped_epoch}/300")
print(f"Train acc finale:    {final_train_acc:.1f}%")
print(f"Écart train-test:    {final_train_acc - test_acc*100:.1f}%")
print("="*70)

val_accs = np.array(history.history['val_accuracy']) * 100
top5_epochs = np.argsort(val_accs)[-5:][::-1]
print("\nTop 5 val_accuracy:")
for i, ep in enumerate(top5_epochs):
    print(f"   #{i+1}: Epoch {ep+1} → {val_accs[ep]:.1f}%")

epochs_above_85 = np.sum(val_accs >= 85.0)
epochs_above_80 = np.sum(val_accs >= 80.0)
print(f"\nEpochs ≥ 85.0%: {epochs_above_85}")
print(f"Epochs ≥ 80.0%: {epochs_above_80}")

print(f"\nCNN: baseline=72.5% → T1=77.5% → T2={test_acc*100:.1f}%")
print(f"vs MLP optimisé: 80.0%")

if test_acc >= 0.85:
    print("ABOVE_85")
else:
    print("On continue d'optimiser #Smile")
print("="*70)
