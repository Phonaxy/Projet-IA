import os

def parse_model_file(filepath):
    """
    Lit le fichier texte contenant les poids du modele (mlp_model.txt)
    et organise les donnees par section dans un dictionnaire.
    """
    weights_data = {}
    current_section = None
    buffer = []
    
    # Vérification de l'existence du fichier
    if not os.path.exists(filepath):
        print(f"Erreur : Le fichier {filepath} est introuvable.")
        return None

    print(f"-> Lecture du fichier modele : {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parcours du fichier ligne par ligne
    for line in lines:
        line = line.strip()
        
        # On ignore les lignes vides et les commentaires
        if not line or line.startswith('#'):
            continue
        
        # Détection d'une nouvelle section (ex: [layer1_weights])
        if line.startswith('[') and line.endswith(']'):
            # Si on était déjà en train de lire une section, on sauvegarde ce qu'on a trouvé
            if current_section:
                weights_data[current_section] = buffer
                print(f"   Section '{current_section}' chargee : {len(buffer)} valeurs.")
                buffer = []
            
            # On récupère le nom de la nouvelle section sans les crochets
            current_section = line[1:-1]
        else:
            # Traitement des données numériques
            parts = line.split()
            
            try:
                # On essaie de convertir chaque bout de texte en nombre décimal (float)
                values = [float(x) for x in parts]
                
                # Petite astuce : parfois la première ligne d'une section contient les dimensions (ex: "784 128")
                # On veut éviter de mettre ces entiers dans notre tableau de poids.
                # Si le buffer est vide et qu'on a peu de valeurs entières, c'est sûrement les dimensions.
                if len(buffer) == 0 and len(values) <= 2 and all(x.is_integer() for x in values):
                    # On ignore cette ligne, on veut juste les données brutes
                    continue
                
                # On ajoute les valeurs trouvées à notre liste temporaire
                buffer.extend(values)
            except ValueError:
                # Si ce n'est pas du texte convertissable en nombre, on passe
                continue

    # Ne pas oublier d'enregistrer la toute dernière section du fichier
    if current_section and buffer:
        weights_data[current_section] = buffer
        print(f"   Section '{current_section}' chargee : {len(buffer)} valeurs.")
        
    return weights_data

def generate_c_header(data, output_path):
    """
    Genere un fichier d'en-tête C (.h) valide contenant les tableaux statiques
    pour les poids et les biais, prets a etre compiles.
    """
    print(f"-> Generation du fichier C header : {output_path}")
    
    # Debut du fichier C
    c_content = """#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

// =================================================================
//  Fichier genere automatiquement par export_weights.py
//  Contient les poids du reseau de neurones pour l'inference C.
//  NE PAS MODIFIER MANUELLEMENT !
// =================================================================

"""
    
    # Pour s'y retrouver, on définit l'architecture attendue : 784 (Input) -> 128 (Hidden) -> 10 (Output)
    
    # 1. Poids de la Couche Cachee (Layer 1)
    if 'layer1_weights' in data:
        w1 = data['layer1_weights']
        c_content += f"// --- Couche 1 : Entree (784) vers Cachee (128) ---\n"
        c_content += f"// Dimensions attendues\n"
        c_content += f"static const int L1_IN_DIM = 784;\n"
        c_content += f"static const int L1_OUT_DIM = 128;\n\n"
        
        c_content += f"// Tableau plat des poids (Matrice 784x128 aplatie)\n"
        c_content += f"static const float LAYER1_WEIGHTS[{len(w1)}] = {{\n    "
        # Astuce d'affichage : on formate les nombres proprement et on les sépare par des virgules
        c_content += ", ".join(f"{x:.8f}f" for x in w1)
        c_content += "\n};\n\n"

    # 2. Biais de la Couche Cachee (Layer 1)
    if 'layer1_bias' in data:
        b1 = data['layer1_bias']
        c_content += f"// Biais de la couche 1 (1 valeur par neurone cache)\n"
        c_content += f"static const float LAYER1_BIAS[{len(b1)}] = {{\n    "
        c_content += ", ".join(f"{x:.8f}f" for x in b1)
        c_content += "\n};\n\n"

    # 3. Poids de la Couche de Sortie (Layer 2)
    if 'layer2_weights' in data:
        w2 = data['layer2_weights']
        c_content += f"// --- Couche 2 : Cachee (128) vers Sortie (10) ---\n"
        c_content += f"static const int L2_IN_DIM = 128;\n"
        c_content += f"static const int L2_OUT_DIM = 10;\n\n"
        
        c_content += f"// Tableau plat des poids (Matrice 128x10 aplatie)\n"
        c_content += f"static const float LAYER2_WEIGHTS[{len(w2)}] = {{\n    "
        c_content += ", ".join(f"{x:.8f}f" for x in w2)
        c_content += "\n};\n\n"

    # 4. Biais de la Couche de Sortie (Layer 2)
    if 'layer2_bias' in data:
        b2 = data['layer2_bias']
        c_content += f"// Biais de la couche 2 (1 valeur par classe de sortie)\n"
        c_content += f"static const float LAYER2_BIAS[{len(b2)}] = {{\n    "
        c_content += ", ".join(f"{x:.8f}f" for x in b2)
        c_content += "\n};\n\n"

    # Fin du fichier C (garde d'inclusion)
    c_content += "#endif // MODEL_WEIGHTS_H\n"

    # Ecriture physique sur le disque
    with open(output_path, 'w') as f:
        f.write(c_content)
    
    print("-> Succes ! Le fichier model_weights.h est pret.")

if __name__ == "__main__":
    # Definition des chemins relatifs
    # On suppose que ce script est dans 'training/' et que le modele est dans 'models/'
    base_dir = os.path.dirname(__file__)
    input_file = os.path.join(base_dir, '../models/mlp_model.txt')
    # Le fichier de sortie ira dans le dossier du code C
    output_file = os.path.join(base_dir, '../inference_c/model_weights.h')
    
    print("=== Export des Poids vers C header ===")
    
    # 1. Analyse du fichier texte
    weights = parse_model_file(input_file)
    
    # 2. Generation du code C si tout va bien
    if weights:
        generate_c_header(weights, output_file)
    else:
        print("Echec : Impossible de recuperer les poids.")
