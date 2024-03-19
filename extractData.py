import cv2
import os
# Directorio donde se encuentran las carpetas con los videos
videos_dir = "DATASET"

# Función para extraer frames de un video
def extract_frames(video_path, output_dir):
    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtener la cantidad de frames totales
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iterar sobre los frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar el frame
        cv2.imshow('Frame', frame)
        
        # Esperar por la tecla
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Presionar 'q' para salir
            break
        elif key == ord('s'):  # Presionar 's' para guardar el frame
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame guardado: {frame_path}")
            frame_count += 1

    # Liberar el video y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

# Iterar sobre las carpetas
for folder in os.listdir(videos_dir):
    folder_path = os.path.join(videos_dir, folder)
    if os.path.isdir(folder_path):
        print(f"Extrayendo frames de la carpeta: {folder}")
        
        # Iterar sobre los videos en la carpeta
        for video_file in os.listdir(folder_path):
            if video_file.endswith(".mp4") or video_file.endswith(".avi"):
                video_path = os.path.join(folder_path, video_file)
                output_dir = os.path.join(folder_path, "frames")
                print(f"Extrayendo frames del video: {video_file}")
                extract_frames(video_path, output_dir)

