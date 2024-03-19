from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pose_media as pm
import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

class Ui_Dialog:
    threshold = 0.8
    actions = np.array(['Falling down', 'Sit down', 'Stand up', 'still sitting', 'Walking'])
    pTime = 0
    cTime = 0

    def __init__(self):
        self.pose = pm.mediapipe_pose()
        self.pt = self.pose.mp_holistic.Holistic()
        self.new_model = tf.keras.models.load_model('ModeloBacano5.h5')
        self.sequence = []
        self.sentence = []

    def detect_actions(self, frame):
        try:
            frame, results = self.pose.mediapipe_detection(frame, self.pt)
        except:
            results = None  # Si ocurre un error, establecer results en None
        if results is not None:  # Verificar si results no es None
            self.pose.draw_styled_landmarks(frame, results)
            keypoints = self.pose.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            if len(self.sequence) == 30:
                res = self.new_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence) > 0:
                        if self.actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(self.actions[np.argmax(res)])
                    else:
                        self.sentence.append(self.actions[np.argmax(res)])
                if len(self.sentence) > 1:
                    self.sentence = self.sentence[-1:]
            return ' '.join(self.sentence)
        else:
            return ''

def video_capture():
    ui = Ui_Dialog()  # Crear una instancia de Ui_Dialog
    cap = cv2.VideoCapture(0)  # Utilizar la interfaz OpenCV para captura de video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:  # Verificar si la lectura del cuadro de video fue exitosa
            text = ui.detect_actions(frame)  # Llamar a detect_actions en la instancia ui
            ui.cTime = time.time()
            fps = 1 / (ui.cTime - ui.pTime)
            ui.pTime = ui.cTime
            frame_list = frame.tolist()  # Convertir frame a lista
            socketio.emit('frame', {'frame': frame_list, 'text': text, 'fps': int(fps)})
        else:
            print("Error: No se pudo capturar el cuadro de video")
            break

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global video_thread
    video_thread = threading.Thread(target=video_capture)
    video_thread.daemon = True
    video_thread.start()

if __name__ == "__main__":
    socketio.run(app)