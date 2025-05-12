#!/bin/bash

# Activar entorno conda
source /opt/conda/etc/profile.d/conda.sh
conda activate detectron2

# Ejecutar el script de Python con los argumentos proporcionados
if [ "$1" == "--help" ]; then
    echo "=== Detector de Objetos con Detectron2 ==="
    echo "Uso: docker run --gpus all -v /ruta/local:/app/data detectron2 --video /app/data/input/video.mp4 --output /app/data/output [--stats]"
    echo ""
    echo "Opciones:"
    echo "  --video RUTA    Ruta al archivo de video (MP4, WAV)"
    echo "  --output DIR    Directorio para guardar resultados"
    echo "  --stats         Opcional, genera estadísticas de detección"
    echo ""
    echo "Ejemplo:"
    echo "  docker run --gpus all -v /home/usuario/videos:/app/data detectron2 --video /app/data/input/mi_video.mp4 --output /app/data/output --stats"
else
    echo "=== Detector de Objetos con Detectron2 ==="
    echo "Ejecutando detector de objetos..."
    python detector_objetos.py "$@"
fi