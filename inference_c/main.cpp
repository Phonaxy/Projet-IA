/**
 * main.cpp
 * Pi 5 Camera Stream + Inference MLP chiffres manuscrits
 * Base : code prof (capture H264 + stream RTSP)
 * Ajout : preprocessing, inference MLP, affichage resultats + stats
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdio>

extern "C" {
#include "neural_network.h"
}

/* Variables globales */
std::atomic<bool> running(true);
MLPModel model;

/* Stats globales pour affichage final */
int g_frame_count = 0;
int g_total_detections = 0;
float g_inference_times[100];
int g_inference_idx = 0;
std::chrono::steady_clock::time_point g_start_time;

void print_final_stats() {
    auto end_time = std::chrono::steady_clock::now();
    float total_elapsed = std::chrono::duration<float>(end_time - g_start_time).count();

    if (g_frame_count == 0) return;

    float final_fps = g_frame_count / total_elapsed;

    /* Temps inference moyen sur les dernieres frames */
    float avg_inference = 0.0f;
    int count = (g_inference_idx < 100) ? g_inference_idx : 100;
    if (count > 0) {
        for (int i = 0; i < count; i++) {
            avg_inference += g_inference_times[i];
        }
        avg_inference /= count;
    }

    printf("\n");
    printf("========================================\n");
    printf("         STATISTIQUES FINALES          \n");
    printf("========================================\n");
    printf("Frames totales:        %d\n", g_frame_count);
    printf("Duree:                 %.1f secondes\n", total_elapsed);
    printf("FPS moyen:             %.2f\n", final_fps);
    printf("Temps inference moyen: %.2f ms\n", avg_inference);
    printf("Detections totales:    %d (%.1f%%)\n",
           g_total_detections,
           (float)g_total_detections / g_frame_count * 100);
    printf("========================================\n");
}

void sigHandler(int) {
    running = false;
}

int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    /* Charger le modele MLP */
    if (load_model("../models/mlp_model.txt", &model) != 0) {
        std::cerr << "Erreur: chargement modele echoue" << std::endl;
        return 1;
    }

    int inPort = argc > 1 ? std::stoi(argv[1]) : 5000;
    int outPort = argc > 2 ? std::stoi(argv[2]) : 8554;
    int w = argc > 3 ? std::stoi(argv[3]) : 1280;
    int h = argc > 4 ? std::stoi(argv[4]) : 720;

    std::cout << "=== Pi5 Camera + MLP ===" << std::endl;
    std::cout << "In:" << inPort << " Out:" << outPort << " " << w << "x" << h << std::endl;

    std::string capPipe =
        "tcpclientsrc host=127.0.0.1 port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe =
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) { std::cerr << "Erreur: input" << std::endl; return 1; }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) { std::cerr << "Erreur: output" << std::endl; return 1; }

    cv::Mat frame;

    /* Buffers reutilisables (pas d'allocation dans la boucle) */
    cv::Mat gray, resized, inverted;
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    g_start_time = std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        if (frame.empty()) continue;

        /* --- Preprocessing --- */
        auto t1 = std::chrono::steady_clock::now();

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, resized, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);

        /* Inversion couleurs (fond blanc -> fond noir comme MNIST) */
        inverted = 255 - resized;

        /* Normalisation [0, 1] */
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input[i * 28 + j] = inverted.at<uchar>(i, j) / 255.0f;
            }
        }

        /* --- Inference MLP --- */
        forward_pass(input, output, &model);

        auto t2 = std::chrono::steady_clock::now();
        float inference_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        /* --- Post-traitement --- */
        int predicted = argmax(output, OUTPUT_SIZE);
        float confidence = output[predicted];

        g_frame_count++;

        /* Affichage si confiance suffisante */
        if (confidence > 0.7f) {
            printf("Frame %05d | Chiffre: %d | Confiance: %.1f%% | Temps: %.2f ms\n",
                   g_frame_count, predicted, confidence * 100.0f, inference_time);
            g_total_detections++;
        }

        /* Stocker temps inference pour stats */
        g_inference_times[g_inference_idx % 100] = inference_time;
        g_inference_idx++;

        /* Stats toutes les 100 frames */
        if (g_frame_count % 100 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(current_time - g_start_time).count();
            float fps = g_frame_count / elapsed;

            float avg_inference = 0.0f;
            int count = (g_inference_idx < 100) ? g_inference_idx : 100;
            for (int i = 0; i < count; i++) {
                avg_inference += g_inference_times[i];
            }
            avg_inference /= count;

            printf("\n========== STATS ==========\n");
            printf("Frames: %d\n", g_frame_count);
            printf("FPS moyen: %.1f\n", fps);
            printf("Inference moyenne: %.2f ms\n", avg_inference);
            printf("Detections: %d (%.1f%%)\n", g_total_detections,
                   (float)g_total_detections / g_frame_count * 100);
            printf("===========================\n\n");
        }

        /* Envoi frame vers stream RTSP */
        writer.write(frame);
    }

    /* Stats finales */
    print_final_stats();

    return 0;
}