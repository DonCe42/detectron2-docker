#!/usr/bin/env python3
"""
Detector de objetos usando Detectron2 para procesar videos
Requisitos:
- Ubuntu
- CUDA instalado
- PyTorch con soporte CUDA
- Detectron2
"""

import argparse
import cv2
import numpy as np
import os
import torch
import time
from datetime import datetime

# Importaciones de Detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def setup_detectron2():
    """Configurar Detectron2 con el modelo Mask R-CNN"""
    from detectron2.model_zoo import model_zoo
    
    cfg = get_cfg()
    # Usar el modelo preentrenado Mask R-CNN con Backbone ResNet50-FPN
    config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Umbral para detecciones
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Configuración de GPU
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("GPU no disponible, usando CPU")
    
    return cfg

def process_video(video_path, output_dir):
    """Procesar video y detectar objetos frame por frame"""
    # Verificar que el video existe
    if not os.path.isfile(video_path):
        print(f"Error: El archivo de video {video_path} no existe.")
        return
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configurar Detectron2
    cfg = setup_detectron2()
    predictor = DefaultPredictor(cfg)
    
    # Obtener la metadata para visualización
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolución: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Configurar el video de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"detections_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Procesar el video frame por frame
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  # Mostrar progreso cada 10 frames
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            print(f"Procesando... {progress:.1f}% ({frame_count}/{total_frames}) - {fps_processing:.2f} FPS")
        
        # Realizar la detección con Detectron2
        outputs = predictor(frame)
        
        # Visualizar los resultados
        v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1.0)
        instances = outputs["instances"].to("cpu")
        vis_output = v.draw_instance_predictions(instances)
        result_frame = vis_output.get_image()[:, :, ::-1]
        
        # Escribir el frame en el video de salida
        out.write(result_frame)
        
        # Opcional: mostrar el frame (comentar para procesamiento más rápido)
        # cv2.imshow('Detectron2 Object Detection', result_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
        #     break
    
    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"Procesamiento completado en {total_time:.2f} segundos")
    print(f"Video guardado en: {output_video_path}")
    
    return output_video_path

def save_detection_stats(video_path, output_dir):
    """Guardar estadísticas de detección en un archivo CSV"""
    # Configurar Detectron2
    cfg = setup_detectron2()
    predictor = DefaultPredictor(cfg)
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    # Preparar archivo CSV para estadísticas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = os.path.join(output_dir, f"detection_stats_{timestamp}.csv")
    with open(stats_path, 'w') as f:
        f.write("frame,class,confidence,x1,y1,x2,y2\n")
    
    frame_count = 0
    object_counts = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Realizar la detección
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        
        # Extraer clases, cajas y confianzas
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            
            # Actualizar conteo de objetos
            for class_id in classes:
                class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
            
            # Guardar detalles en CSV
            with open(stats_path, 'a') as f:
                for i in range(len(classes)):
                    class_id = classes[i]
                    class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
                    box = boxes[i]
                    score = scores[i]
                    f.write(f"{frame_count},{class_name},{score:.4f},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}\n")
    
    cap.release()
    
    # Guardar resumen de objetos detectados
    summary_path = os.path.join(output_dir, f"detection_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=== Resumen de Detección de Objetos ===\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total de frames procesados: {frame_count}\n\n")
        f.write("Objetos detectados:\n")
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {obj}: {count}\n")
    
    print(f"Estadísticas guardadas en: {stats_path}")
    print(f"Resumen guardado en: {summary_path}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Detectron2 Object Detection para videos')
    parser.add_argument('--video', required=True, help='Ruta al archivo de video (MP4, WAV)')
    parser.add_argument('--output', default='output', help='Directorio para guardar resultados')
    parser.add_argument('--stats', action='store_true', help='Generar estadísticas de detección')
    
    args = parser.parse_args()
    
    # Verificar que el video tenga un formato válido
    if not args.video.lower().endswith(('.mp4', '.wav')):
        print("Error: El formato de video debe ser MP4 o WAV")
        return
    
    # Procesar el video
    processed_video = process_video(args.video, args.output)
    
    # Opcionalmente generar estadísticas
    if args.stats and processed_video:
        save_detection_stats(args.video, args.output)

if __name__ == "__main__":
    print("=== Detector de Objetos con Detectron2 ===")
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecución: {e}")