FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Evitar preguntas interactivas durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Configurar variables de entorno
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libgl1-mesa-glx \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Crear entorno conda
RUN conda create -n detectron2 python=3.10 -y

# Configurar shell para usar comandos conda
SHELL ["/bin/bash", "-c"]

# Activar entorno y instalar dependencias (corregido con source)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate detectron2 && \
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    conda install -c conda-forge opencv cython -y && \
    pip install pycocotools && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copiar los archivos de la aplicación
WORKDIR /app
COPY detector_objetos.py /app/

# Script de inicialización
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Crear un directorio para la entrada/salida de archivos
RUN mkdir -p /app/data/input /app/data/output

# Exponer un puerto por si se añade una interfaz web en el futuro
EXPOSE 8000

# Usar el entrypoint para activar el entorno conda
ENTRYPOINT ["/app/entrypoint.sh"]

# Comando por defecto (se puede sobreescribir al ejecutar el contenedor)
CMD ["--help"]