import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path pour importer utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports des modules du projet
from utils import benchmark, visualization

# ============================================================================
# 1. DEFINITIONS (Duplication pour ne pas toucher aux scripts d'entraînement)
# ============================================================================

def load_custom_dataset(data_dir="/home/docker/Work/data/custom_digits"):
    """Charge le dataset custom depuis les dossiers 0-9"""
    data_dir = Path(data_dir)
    
    # Tentative avec chemin relatif si le chemin absolu échoue (pour compatibilité Windows/Docker)
    if not data_dir.exists():
        # Essayer un chemin relatif standard
        local_path = Path("data/custom_digits")
        if local_path.exists():
             data_dir = local_path
        # Essayer data/custom_digits_inverted pour le CNN si besoin (on prendra le même pour les deux ici pour l'équité)
        elif Path("data/custom_digits_inverted").exists():
             data_dir = Path("data/custom_digits_inverted")
    
    if not data_dir.exists():
        print(f"  Attention: Dossier de données introuvable ({data_dir}).")
        return (np.array([]), np.array([])), (np.array([]), np.array([]))
        
    images = []
    labels = []
    
    # Charger chaque chiffre (0-9)
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        if not digit_dir.exists(): continue
        
        # Charger toutes les images BMP
        for img_path in sorted(digit_dir.glob("*.bmp")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                labels.append(digit)
    
    if not images:
        return (np.array([]), np.array([])), (np.array([]), np.array([]))

    X = np.array(images, dtype='float32')
    y = np.array(labels)
    
    # Split train/test (80/20)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (x_train, y_train), (x_test, y_test)

def create_mlp_model(input_shape=(28, 28)):
    """Crée et compile le modèle MLP (Identique à train_mlp.py)"""
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_model(input_shape=(28, 28, 1)):
    """Crée et compile le modèle CNN (Identique à train_cnn.py)"""
    model = keras.Sequential([
        # Bloc Conv 1 : 28×28×1 → 14×14×16
        layers.Conv2D(16, 3, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001),
                      input_shape=input_shape),
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
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_or_train_model(model_type, x_train, y_train, x_test, y_test):
    """
    Charge le modèle s'il existe, sinon l'entraîne et le sauvegarde.
    """
    params_dir = Path("models")
    params_dir.mkdir(exist_ok=True)
    model_path = params_dir / f"{model_type}_model.keras"
    
    if model_type == 'mlp':
        model = create_mlp_model()
        epochs = 50
    else:
        model = create_cnn_model()
        epochs = 100 # Un peu moins que 300 pour le test
    
    if model_path.exists():
        print(f"    Chargement du modèle {model_type.upper()} existant...")
        try:
            model.load_weights(model_path)
            # Petite validation
            model.evaluate(x_test[:5], y_test[:5], verbose=0)
            return model
        except Exception as e:
            print(f"    Erreur chargement: {e}. Nouvel entraînement requis.")
    
    print(f"    Entraînement du modèle {model_type.upper()} ({epochs} epochs)...")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    # Adaptation des données pour le CNN
    x_t = x_train
    x_v = x_test
    if model_type == 'cnn':
        x_t = x_train.reshape(-1, 28, 28, 1)
        x_v = x_test.reshape(-1, 28, 28, 1)
        
    model.fit(
        x_t, y_train,
        epochs=epochs,
        validation_data=(x_v, y_test),
        callbacks=callbacks,
        verbose=1,
        batch_size=16 if model_type=='cnn' else 8
    )
    
    print(f"    Sauvegarde du modèle dans {model_path}")
    model.save(model_path)
    
    return model

# ============================================================================
# 2. MAIN EXECUTION
# ============================================================================

def main():
    print("===========================================================")
    print(" COMPARATEUR DE MODÈLES : MLP vs CNN")
    print("===========================================================")

    # 1. Charger les données
    print("\n[1/5] Chargement du Dataset...")
    (x_train, y_train), (x_test, y_test) = load_custom_dataset()
    
    if len(x_test) == 0:
        print("❌ Erreur : Dataset vide ou introuvable. Verifiez 'data/custom_digits' ou 'data/custom_digits_inverted'.")
        return

    # Normalisation
    x_train_norm = x_train / 255.0
    x_test_norm = x_test / 255.0
    
    print(f" Données chargées : {len(x_train)} train, {len(x_test)} test")

    # 2. Préparer les modèles
    print("\n[2/5] Préparation des Modèles...")
    
    # --- MLP ---
    print("-> MLP...")
    mlp = get_or_train_model('mlp', x_train_norm, y_train, x_test_norm, y_test)

    # --- CNN ---
    print("-> CNN...")
    cnn = get_or_train_model('cnn', x_train_norm, y_train, x_test_norm, y_test)

    # 3. Benchmark Technique (Taille & Vitesse)
    print("\n[3/5] Benchmark Technique...")
    
    print("\n--- MLP Metrics ---")
    mlp_size_mb, mlp_params = benchmark.get_model_size_mb(mlp, print_summary=False)
    mlp_time_ms = benchmark.measure_inference_time(mlp, x_test_norm[0])

    print("\n--- CNN Metrics ---")
    cnn_size_mb, cnn_params = benchmark.get_model_size_mb(cnn, print_summary=False)
    # Pour le CNN, l'entrée doit être (28, 28, 1)
    x_test_cnn = x_test_norm.reshape(-1, 28, 28, 1)
    cnn_time_ms = benchmark.measure_inference_time(cnn, x_test_cnn[0])

    # 4. Benchmark Performances (Précision)
    print("\n[4/5] Benchmark Précision...")
    
    print("\n--- Évaluation MLP ---")
    mlp_metrics = benchmark.evaluate_model_performance(mlp, x_test_norm, y_test)
    
    print("\n--- Évaluation CNN ---")
    cnn_metrics = benchmark.evaluate_model_performance(cnn, x_test_cnn, y_test)

    # 5. Visualisation & Rapport
    print("\n[5/5] Génération des Graphiques...")
    output_dir = Path("docs/benchmark_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # A. Comparaison Accuracy
    accuracies = {'MLP': mlp_metrics['accuracy'], 'CNN': cnn_metrics['accuracy']}
    visualization.plot_model_comparison(accuracies, metric_name='Accuracy', 
                                      title='Comparaison Précision (Test Set)', 
                                      save_path=output_dir / 'comparison_accuracy.png')

    # B. Comparaison Temps Inférence
    times = {'MLP': mlp_time_ms, 'CNN': cnn_time_ms}
    visualization.plot_inference_time_comparison(times, 
                                               title="Temps d'Inférence (ms/sample)", 
                                               save_path=output_dir / 'comparison_time.png')

    # C. Comparaison Taille (Paramètres)
    params = {'MLP': mlp_params, 'CNN': cnn_params}
    visualization.plot_model_comparison(params, metric_name='Nombre de Paramètres', 
                                      title='Comparaison Complexité (Nb Paramètres)', 
                                      save_path=output_dir / 'comparison_params.png')

    # D. Matrices de Confusion
    classes = [str(i) for i in range(10)]
    visualization.plot_confusion_matrix(y_test, mlp_metrics['y_pred'], classes, 
                                      title="Matrice de Confusion MLP", 
                                      save_path=output_dir / 'confusion_matrix_mlp.png')
    
    visualization.plot_confusion_matrix(y_test, cnn_metrics['y_pred'], classes, 
                                      title="Matrice de Confusion CNN", 
                                      save_path=output_dir / 'confusion_matrix_cnn.png')

    # E. Exemples d'erreurs CNN (pour analyse)
    errors_idx = np.where(cnn_metrics['y_pred'] != y_test)[0]
    if len(errors_idx) > 0:
        print(f"-> Génération d'exemples d'erreurs CNN ({len(errors_idx)} erreurs trouvées)...")
        sample_indices = np.random.choice(errors_idx, min(10, len(errors_idx)), replace=False)
        visualization.plot_predictions_sample(
            x_test[sample_indices], 
            y_test[sample_indices], 
            cnn_metrics['y_pred'][sample_indices], 
            classes, 
            title="Erreurs du CNN", 
            save_path=output_dir / 'cnn_errors_sample.png'
        )
    
    print(f"\n Terminé ! Les résultats sont dans : {output_dir.absolute()}")
    print("="*70)
    print("RÉSUMÉ FINAL :")
    print(f"MLP -> Acc: {mlp_metrics['accuracy']:.2%} | Time: {mlp_time_ms:.4f}ms | Params: {mlp_params:,}")
    print(f"CNN -> Acc: {cnn_metrics['accuracy']:.2%} | Time: {cnn_time_ms:.4f}ms | Params: {cnn_params:,}")
    print("="*70)

if __name__ == "__main__":
    main()
