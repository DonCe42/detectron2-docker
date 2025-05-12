# Detector de Objetos con Detectron2 en Docker

Este proyecto proporciona una aplicación de detección de objetos en video utilizando Detectron2, empaquetada en un contenedor Docker para facilitar su despliegue y uso en cualquier máquina o en la nube.

## Requisitos

- Docker instalado ([Instrucciones de instalación](https://docs.docker.com/get-docker/))
- Para uso con GPU:
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Controladores NVIDIA actualizados

## Estructura del Proyecto

```
detectron2-docker/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── detector_objetos.py
├── data/
│   ├── input/
│   └── output/
└── README.md
```

## Instrucciones de Uso

### 1. Preparación

1. Clona o descarga este repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd detectron2-docker
   ```

2. Coloca tus videos para análisis en la carpeta `data/input/`

### 2. Construir la Imagen Docker

```bash
docker build -t detectron2-app .
```

O usando docker-compose:

```bash
docker-compose build
```

### 3. Ejecutar el Detector de Objetos

#### Usando Docker directamente:

```bash
# Con GPU (recomendado)
docker run --gpus all -v $(pwd)/data:/app/data detectron2-app --video /app/data/input/tu_video.mp4 --output /app/data/output --stats

# Sin GPU
docker run -v $(pwd)/data:/app/data detectron2-app --video /app/data/input/tu_video.mp4 --output /app/data/output --stats
```

#### Usando Docker Compose:

```bash
# Edita docker-compose.yml para establecer los parámetros correctos
docker-compose up
```

### 4. Opciones Disponibles

```
--video RUTA    Ruta al archivo de video (MP4, WAV)
--output DIR    Directorio para guardar resultados
--stats         Opcional, genera estadísticas de detección
```

## Despliegue en la Nube

### AWS

1. Instala AWS CLI y configura tus credenciales.
2. Sube tu imagen a Amazon ECR:

```bash
# Autenticación en ECR
aws ecr get-login-password --region <tu-region> | docker login --username AWS --password-stdin <tu-cuenta>.dkr.ecr.<tu-region>.amazonaws.com

# Crea un repositorio
aws ecr create-repository --repository-name detectron2-app --region <tu-region>

# Etiqueta y sube la imagen
docker tag detectron2-app:latest <tu-cuenta>.dkr.ecr.<tu-region>.amazonaws.com/detectron2-app:latest
docker push <tu-cuenta>.dkr.ecr.<tu-region>.amazonaws.com/detectron2-app:latest
```

3. Ejecuta el contenedor en EC2 con GPU o en servicios como AWS Batch.

### Google Cloud

1. Configura Google Cloud SDK.
2. Sube tu imagen a Google Container Registry:

```bash
# Configura docker para GCR
gcloud auth configure-docker

# Etiqueta y sube la imagen
docker tag detectron2-app:latest gcr.io/<tu-proyecto>/detectron2-app:latest
docker push gcr.io/<tu-proyecto>/detectron2-app:latest
```

3. Ejecuta el contenedor en GCE con GPUs o en Google Kubernetes Engine.

### Azure

1. Configura Azure CLI.
2. Sube tu imagen a Azure Container Registry:

```bash
# Inicia sesión en Azure
az login

# Crea un registro de contenedores (si no existe)
az acr create --resource-group <tu-grupo-recursos> --name <tu-registro> --sku Basic

# Inicia sesión en el registro
az acr login --name <tu-registro>

# Etiqueta y sube la imagen
docker tag detectron2-app:latest <tu-registro>.azurecr.io/detectron2-app:latest
docker push <tu-registro>.azurecr.io/detectron2-app:latest
```

3. Ejecuta el contenedor en Azure Container Instances o Azure Kubernetes Service con soporte para GPUs.

## Solución de Problemas

### Error de CUDA

Si encuentras errores relacionados con CUDA, verifica:

1. Que los controladores NVIDIA están correctamente instalados
2. Que el NVIDIA Container Toolkit está instalado y configurado
3. Que estás usando `--gpus all` al ejecutar el contenedor

### Problemas de Memoria

Si el contenedor se detiene debido a problemas de memoria:

1. Usa videos de menor resolución
2. Ajusta el script para procesar menos frames por segundo
3. Aumenta la memoria asignada al contenedor Docker

## Recursos Adicionales

- [Documentación de Detectron2](https://detectron2.readthedocs.io/)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)