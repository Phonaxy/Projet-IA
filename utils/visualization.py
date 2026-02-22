import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    """
    Affiche et sauvegarde la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé : {save_path}")
    plt.close()

def plot_model_comparison(metrics_dict, metric_name='Accuracy', title='Comparaison des Modèles', save_path=None):
    """
    Affiche un bar chart comparant une métrique pour plusieurs modèles.
    metrics_dict: dict { 'ModelName': value, ... }
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'])
    
    plt.title(title)
    plt.ylabel(metric_name)
    plt.ylim(0, max(values) * 1.1)  # Un peu de marge en haut
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé : {save_path}")
    plt.close()

def plot_inference_time_comparison(times_dict, title='Temps d\'Inférence Moyen (ms)', save_path=None):
    """
    Affiche un bar chart comparant les temps d'inférence.
    """
    plot_model_comparison(times_dict, metric_name='Temps (ms)', title=title, save_path=save_path)

def plot_predictions_sample(images, true_labels, pred_labels, classes, title="Exemples de Prédictions", save_path=None):
    """
    Affiche une grille d'images avec leurs prédictions vs réalité.
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        label_text = f"True: {classes[true_labels[i]]}\nPred: {classes[pred_labels[i]]}"
        
        plt.title(label_text, color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé : {save_path}")
    plt.close()