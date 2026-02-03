# Makefile global pour le projet IA Embarquée
# Sert de raccourci pour les commandes courantes

# Variables
DOCKER_COMPOSE = docker-compose
CONTAINER_NAME = projetia_env
WORK_DIR = /home/docker/Work

.PHONY: help build up down shell train-mlp export compile-c clean status logs

# === Aide ===
help:
	@echo "Commandes disponibles pour le projet :"
	@echo "--------------------------------------------------------"
	@echo "  make build       : Construit/Reconstruit l'image Docker"
	@echo "  make up          : Lance le conteneur en arrière-plan"
	@echo "  make down        : Arrête et supprime le conteneur"
	@echo "  make shell       : Accède au terminal du conteneur"
	@echo "  make status      : Affiche l'état du conteneur"
	@echo "  make logs        : Affiche les logs du conteneur"
	@echo "--------------------------------------------------------"
	@echo "  make train-mlp   : Lance l'entrainement du MLP (dans Docker)"
	@echo "  make export      : Exporte les poids du modèle (dans Docker)"
	@echo "  make compile-c   : Compile le code C d'inférence (dans Docker)"
	@echo "--------------------------------------------------------"
	@echo "  make clean       : Nettoie les fichiers temporaires et builds"

# === Gestion Docker ===
build:
	@echo "Construction de l'image Docker..."
	$(DOCKER_COMPOSE) build

up:
	@echo "Démarrage du conteneur..."
	$(DOCKER_COMPOSE) up -d

down:
	@echo "Arrêt du conteneur..."
	$(DOCKER_COMPOSE) down

shell:
	@echo "Connexion au conteneur..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

status:
	docker ps -f name=$(CONTAINER_NAME)

logs:
	docker logs -f $(CONTAINER_NAME)

# === Raccourcis pour le Workflow IA (exécutés DANS le conteneur) ===
train-mlp:
	@echo "Lancement de l'entraînement MLP..."
	docker exec -it $(CONTAINER_NAME) python3 training/train_mlp.py

export:
	@echo "Export des poids..."
	docker exec -it $(CONTAINER_NAME) python3 training/export_weights.py

compile-c:
	@echo "Compilation du code C..."
	docker exec -it $(CONTAINER_NAME) make -C inference_c

# === Nettoyage ===
clean:
	@echo "Nettoyage des fichiers temporaires..."
	# Nettoyage Python
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	# Nettoyage Compilation C
	cd inference_c && make clean 2>/dev/null || true
