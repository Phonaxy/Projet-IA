import time
import numpy as np
import tensorflow as tf

def get_model_size_mb(model, print_summary=True):
    """
    Estime la taille mémoire d'un modèle Keras (poids + structure).
    """
    if print_summary:
        model.summary()
        
    # Compter les paramètres totaux
    total_params = model.count_params()
    # Chaque paramètre float32 prend 4 octets
    
    # Estimation simple (poids seulement)
    memory_usage = total_params * 4
    memory_usage_mb = memory_usage / (1024 * 1024)
    
    print(f"-> Paramètres totaux : {total_params:,}")
    print(f"-> Taille estimée (poids float32) : {memory_usage_mb:.2f} MB")
    
    return memory_usage_mb, total_params

def measure_inference_time(model, sample_input, num_iterations=1000):
    """
    Mesure le temps moyen d'inférence par échantillon.
    Force un warm-up avant la mesure.
    """
    # S'assurer que l'input a la bonne forme (batch size = 1 pour test unitaire)
    if len(sample_input.shape) == len(model.input_shape) - 1:
        # Ajouter dimension batch si manquant (ex: (28,28) -> (1, 28, 28))
        sample_input = np.expand_dims(sample_input, axis=0)

    # Warm-up (essentiel pour TF/GPU)
    print("-> Warm-up GPU/CPU...")
    for _ in range(10):
        _ = model.predict(sample_input, verbose=0)
    
    print(f"-> Mesure sur {num_iterations} itérations...")
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(sample_input, training=False) # Utiliser __call__ est souvent plus rapide que predict pour un seul item car moins d'overhead
    end_time = time.time()
    
    avg_time_sec = (end_time - start_time) / num_iterations
    avg_time_ms = avg_time_sec * 1000
    
    print(f"-> Temps moyen par inférence : {avg_time_ms:.4f} ms")
    return avg_time_ms

def evaluate_model_performance(model, x_test, y_test):
    """
    Évalue le modèle et retourne un dictionnaire de métriques.
    """
    print("-> Évaluation globale sur le test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    from sklearn.metrics import recall_score, precision_score, f1_score
    
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_test,
        'y_pred': y_pred
    }
    
    print(f"-> Accuracy : {accuracy:.4f}")
    print(f"-> F1 Score : {f1:.4f}")
    
    return metrics
