import cv2
import imageio

# Nombre del archivo GIF de salida
output_gif = 'output.gif'

# Crear una lista para almacenar los frames
frames = []

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Leer los frames de la cámara y agregarlos a la lista
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
    
    # Mostrar el frame en una ventana
    cv2.imshow('Camera', frame)

    
    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

# Guardar los frames como un archivo GIF utilizando ImageIO
imageio.mimsave(output_gif, frames,'GIF', duration=1, fps=30)

print('GIF creado exitosamente.')