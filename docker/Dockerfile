# Dockerfile minimaliste pour projet IA embarquée - MNIST
# TensorFlow uniquement, strict nécessaire

FROM ubuntu:24.04

ARG UID=1000
ARG GID=1000

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Optimisations TensorFlow
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Nettoyage utilisateurs existants
RUN userdel -r $(id -un 1000) 2>/dev/null || true

# ============================================================================
# PACKAGES SYSTÈME ESSENTIELS
# ============================================================================
RUN apt-get update -yq && \
    apt-get install -y --no-install-recommends \
    # Système de base
    sudo nano git wget \
    # Python
    python3 python3-pip python3-venv \
    # Compilation C (pour inférence)
    build-essential cmake gcc g++ \
    # OpenCV et traitement d'image
    libopencv-dev python3-opencv \
    # Export modèles
    libhdf5-dev \
    # Mathématiques
    libatlas-base-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================================
# CONFIGURATION UTILISATEUR
# ============================================================================
RUN mkdir -p "/home/docker" && \
    echo "docker:x:${UID}:${GID}:Docker User,,,:/home/docker:/bin/bash" >> /etc/passwd && \
    echo "docker:x:${UID}:" >> /etc/group && \
    echo "docker:docker" | chpasswd && \
    mkdir -p /etc/sudoers.d && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/docker && \
    chmod 0440 /etc/sudoers.d/docker && \
    chown ${UID}:${GID} -R /home/docker

WORKDIR /home/docker

# ============================================================================
# ENVIRONNEMENT PYTHON TENSORFLOW
# ============================================================================
RUN python3 -m venv /home/docker/venv && \
    . /home/docker/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    # TensorFlow et Keras
    pip install --no-cache-dir tensorflow keras && \
    # Bibliothèques essentielles
    pip install --no-cache-dir numpy matplotlib pandas && \
    pip install --no-cache-dir opencv-python Pillow && \
    # Export modèles
    pip install --no-cache-dir h5py && \
    # Visualisation
    pip install --no-cache-dir scikit-learn seaborn && \
    deactivate

# ============================================================================
# CONFIGURATION BASHRC
# ============================================================================
RUN echo "source /home/docker/venv/bin/activate" >> /home/docker/.bashrc && \
    echo 'echo "Environnement TensorFlow activé"' >> /home/docker/.bashrc && \
    echo 'alias ll="ls -lah"' >> /home/docker/.bashrc && \
    chown ${UID}:${GID} /home/docker/.bashrc

# ============================================================================
# STRUCTURE PROJET
# ============================================================================
RUN mkdir -p /home/docker/Work/projet_mnist/{training,models,inference_c,data/custom_digits} && \
    chown -R ${UID}:${GID} /home/docker

USER docker
WORKDIR /home/docker/Work

CMD ["/bin/bash"]