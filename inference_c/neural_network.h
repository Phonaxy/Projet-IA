/**
 * neural_network.h
 * Inference MLP pour reconnaissance de chiffres manuscrits (0-9)
 * Architecture : 784 -> Dense(128, ReLU) -> Dense(10, Softmax)
 */
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

/* Dimensions du reseau */
#define INPUT_SIZE  784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

/* Structure contenant tous les poids du MLP */
typedef struct {
    float weights_layer1[INPUT_SIZE][HIDDEN_SIZE];   /* 784x128 */
    float bias_layer1[HIDDEN_SIZE];                   /* 128     */
    float weights_layer2[HIDDEN_SIZE][OUTPUT_SIZE];   /* 128x10  */
    float bias_layer2[OUTPUT_SIZE];                   /* 10      */
} MLPModel;

/* Charge les poids depuis un fichier texte. Retourne 0 si OK, -1 si erreur. */
int load_model(const char* filepath, MLPModel* model);

/* Forward pass : calcule les probabilites de sortie a partir d'une image 28x28 */
void forward_pass(const float* input, float* output, const MLPModel* model);

/* Retourne l'indice du maximum dans un tableau */
int argmax(const float* array, int size);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_NETWORK_H */