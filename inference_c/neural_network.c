/**
 * neural_network.c
 * Implementation du forward pass MLP en C pur
 * Pas de dependances externes (juste stdio, stdlib, string, math)
 */
#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Charge les poids du modele depuis un fichier texte structure */
int load_model(const char* filepath, MLPModel* model) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("Erreur: impossible d'ouvrir %s\n", filepath);
        return -1;
    }

    char line[16384];   /* Buffer large pour lignes de 128 floats */
    char section[64];
    int row = 0;
    int dims_read = 0;  /* Flag : dimensions deja lues pour la section */
    int expected_rows = 0;

    section[0] = '\0';

    while (fgets(line, sizeof(line), f)) {
        /* Ignorer lignes vides et commentaires */
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\r')
            continue;

        /* Detection de section */
        if (line[0] == '[') {
            sscanf(line, "[%63[^]]]", section);
            row = 0;
            dims_read = 0;
            continue;
        }

        /* Lecture des dimensions (premiere ligne non-section) */
        if (!dims_read) {
            dims_read = 1;
            /* On ignore les dimensions, on connait l'architecture */
            /* Mais on les lit pour avancer dans le fichier */
            if (strcmp(section, "layer1_weights") == 0) {
                expected_rows = INPUT_SIZE;
            } else if (strcmp(section, "layer2_weights") == 0) {
                expected_rows = HIDDEN_SIZE;
            } else {
                expected_rows = 0; /* bias : une seule ligne */
            }
            continue;
        }

        /* Parsing des poids selon la section */
        if (strcmp(section, "layer1_weights") == 0) {
            if (row >= INPUT_SIZE) continue;
            char* tok = strtok(line, " \t\n\r");
            for (int j = 0; j < HIDDEN_SIZE && tok; j++) {
                model->weights_layer1[row][j] = strtof(tok, NULL);
                tok = strtok(NULL, " \t\n\r");
            }
            row++;
        }
        else if (strcmp(section, "layer1_bias") == 0) {
            char* tok = strtok(line, " \t\n\r");
            for (int j = 0; j < HIDDEN_SIZE && tok; j++) {
                model->bias_layer1[j] = strtof(tok, NULL);
                tok = strtok(NULL, " \t\n\r");
            }
        }
        else if (strcmp(section, "layer2_weights") == 0) {
            if (row >= HIDDEN_SIZE) continue;
            char* tok = strtok(line, " \t\n\r");
            for (int j = 0; j < OUTPUT_SIZE && tok; j++) {
                model->weights_layer2[row][j] = strtof(tok, NULL);
                tok = strtok(NULL, " \t\n\r");
            }
            row++;
        }
        else if (strcmp(section, "layer2_bias") == 0) {
            char* tok = strtok(line, " \t\n\r");
            for (int j = 0; j < OUTPUT_SIZE && tok; j++) {
                model->bias_layer2[j] = strtof(tok, NULL);
                tok = strtok(NULL, " \t\n\r");
            }
        }
    }

    fclose(f);
    printf("Modele charge: %s\n", filepath);
    return 0;
}

/* Forward pass du MLP : input[784] -> output[10] (probabilites) */
void forward_pass(const float* input, float* output, const MLPModel* model) {
    float hidden[HIDDEN_SIZE];

    /* Couche 1 : Dense(128) + ReLU */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = model->bias_layer1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * model->weights_layer1[j][i];
        }
        /* ReLU */
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* Couche 2 : Dense(10) */
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = model->bias_layer2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden[j] * model->weights_layer2[j][i];
        }
        output[i] = sum;
    }

    /* Softmax stable numeriquement */
    float max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) max_val = output[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = expf(output[i] - max_val);
        sum_exp += output[i];
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sum_exp;
    }
}

/* Retourne l'indice du maximum */
int argmax(const float* array, int size) {
    int max_idx = 0;
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }
    return max_idx;
}