import mediapipe as mp
import pose_media as pm
import numpy as np
import cv2
import tensorflow as tf
import customtkinter
from PIL import Image, ImageTk
from threading import Thread
from queue import Queue
import time
import imageio
import requests
import webbrowser

class AppCamera:
    threshold = 0.5
    actions = np.array(['Alerta de Caída',
                        'Normal',
                        'Normal',
                        'Sentándose',
                        'Levantándose',
                        'Sentado',
                        'Caminando'])
    REMOTE_HOST = '3.133.157.169'
# Puerto en el que se ejecuta el servidor Flask
    REMOTE_PORT = 80
    # URL para la recepción de datos en el servidor Flask
    RECEIVE_URL = f"http://{REMOTE_HOST}:{REMOTE_PORT}/receive_data"

    RECEIVE_GIF_URL = f"http://{REMOTE_HOST}:{REMOTE_PORT}/upload"

    RECEIVE_URL_EMAIL = f"http://{REMOTE_HOST}:{REMOTE_PORT}/enviar_email"

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        

        self.q1 = Queue()
        self.q2 = Queue()
        self.q3 = Queue()
        self.q4 = Queue()

        self.fps = 0
        self.pose = pm.mediapipe_pose()
        self.pt = self.pose.mp_holistic.Holistic()
        self.new_model = tf.keras.models.load_model('TrainedModel/ModeloBacano6.h5')
        self.prev_action =  None
        self.Visual = False


        

        self.canvas = customtkinter.CTkCanvas(window, width=800, height=600, bg="#242424", highlightthickness=0)
        self.canvas.pack(side="left")
        
        self.ImBg = cv2.imread('Fondo (2).png')
        self.ImBg = cv2.resize(self.ImBg, (600, 600))
        self.Img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.ImBg, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(100, 0, image=self.Img, anchor="nw")
        

        self.labeltext = customtkinter.CTkLabel(window, text="IP Camera link (http:/..): ")
        self.labeltext.pack(pady=20)


        self.tetxbox = customtkinter.CTkTextbox(window, height=10)
        self.tetxbox.pack(pady=10)

        self.label = customtkinter.CTkLabel(window, fg_color="transparent", text='Acciones: ')
        self.label.pack(pady=40)

        self.button = customtkinter.CTkButton(window, text="Start", command=self.camera)
        self.button.pack(pady=10)

        self.button_draw = customtkinter.CTkButton(window, text="Drawing off", command=self.drawing)
        self.button_draw.pack(pady=10)

        self.button_visual = customtkinter.CTkButton(window, text="Visual on", command=self.VisualB)
        self.button_visual.pack(pady=10)

        self.web_button = customtkinter.CTkButton(window, text="Go to Webpage", command=self.open_webpage)
        self.web_button.pack(pady=10)

        self.notif = customtkinter.CTkLabel(window, text="Notificaciones: ")
        self.notif.pack(pady=40)
        
        
        

        self.sequence = []
        self.sentence = []

        self.GIF = []
        self.draw = False
        self.window.mainloop()

    def drawing(self):
        self.draw = not(self.draw)
        if self.button_draw.cget("text") == "Drawing on":
            self.button_draw.configure(text="Drawing off")
        else:
            self.button_draw.configure(text="Drawing on")
    def open_webpage(self):
        webbrowser.open('http://'+self.REMOTE_HOST+':'+str(self.REMOTE_PORT))

    def VisualB(self):
        self.Visual = not(self.Visual)
        if self.button_visual.cget("text") == "Visual on":
            self.button_visual.configure(text="Visual off")
        else:
            self.button_visual.configure(text="Visual on")
            self.canvas.create_image(100, 0, image=self.Img, anchor="nw")
            self.canvas.image = self.Img  # Store a reference to the image to prevent it from being garbage collected
            self.canvas.configure(bg="#242424")

    def camera(self):
        def processing():
            ip_camera =  self.tetxbox.get(index1=1.0, index2=customtkinter.END).strip()
            cap = cv2.VideoCapture(0) if ip_camera == "0" else cv2.VideoCapture(f"{ip_camera}")
            alert_detected = False
            consecutive = 0
            pTime = 0
            cTime = 0
            while cap.isOpened():
                ret, frameVisual = cap.read()
                frameVisual = cv2.resize(frameVisual,(800,600))
                frame = cv2.resize(frameVisual, (int(frameVisual.shape[1] / 8), int(frameVisual.shape[0] / 8)))
                if not ret:
                    break
                try:
                    frame, results = self.pose.mediapipe_detection(frame, self.pt)
                except:
                    continue
                if self.draw:
                    self.pose.draw_styled_landmarks(frameVisual, results)
                keypoints = self.pose.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    res = self.new_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    if res[np.argmax(res)] > self.threshold:
                        action = self.actions[np.argmax(res)]
                        frameV = cv2.cvtColor(frameVisual, cv2.COLOR_BGR2RGB) 
                        self.GIF.append(frameV)
                        if len(self.GIF) >= 30:
                            self.GIF = self.GIF[-30:]
                        if action == self.prev_action and action != 'Normal':
                            self.prev_action = action
                            consecutive += 1
                            if consecutive == 5:
                                Thread(target=self.sendAction, args=(action,), daemon=True).start()
                                if action == 'Alerta de Caída':
                                    Thread(target=self.createGIF, daemon=True).start()
                        else:
                            self.prev_action = action
                            consecutive = 1
                            self.notif.configure(text="Monitorizando...")
                        if len(self.sentence) == 0 or action != self.sentence[-1]:
                            self.sentence.append(action)
                        self.sentence = self.sentence[-1:]
                text = ' '.join(self.sentence)
                cTime = time.time()
                self.fps = 1 / (cTime - pTime)
                pTime = cTime
                self.label.configure(text=text + f' FPS: {int(self.fps)}')
                
                if not self.Visual:
                    self.canvas.delete("all")
                    self.canvas.create_image(100, 0, image=self.Img, anchor="nw")
                    self.canvas.image = self.Img
                else:
                    frame = cv2.putText(frameVisual, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frameVisual, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=photo, anchor=customtkinter.NW)
                    self.canvas.image = photo  # Store a reference to the image to prevent it from being garbage collected
                
        Thread(target=processing, daemon=True).start()


    def saveFrame(self, frame):
        frameV = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        self.GIF.append(frameV)
        if len(self.GIF) >= 30:
            self.GIF = self.GIF[-30:]

    def createGIF(self):
        self.notif.configure(text="Creando GIF...")
        imageio.mimwrite('Caida2.gif', self.GIF, 'GIF', duration=1, fps=10) 
        print('GIF guardado')
        Thread(target=self.sendAlert, daemon=True).start()
        

    def sendAlert(self):
            try:
                response = requests.post(self.RECEIVE_GIF_URL, files={'file': open('Caida2.gif', 'rb')})
                if response.status_code == 200:
                    print("Alerta enviada correctamente al servidor.")
                    self.notif.configure(text="Alerta enviada \n correctamente al servidor.")
                else:
                    print("Error al enviar la alerta al servidor. Código de estado:", response.status_code)
            except Exception as e:
                print("Error al enviar la alerta al servidor:", str(e))

    def sendAction(self, action):
        try:
            fecha_a = time.time()
            fecha_l = time.localtime(fecha_a)
        
            data = {'action': action, 'date': time.strftime("%Y-%m-%d %H:%M:%S",fecha_l)}
            response = requests.post(self.RECEIVE_URL, json=data)
            response_email = requests.post(self.RECEIVE_URL_EMAIL, json=data)
            if response_email.status_code == 200:
                print("Acción predicha enviada correctamente al servidor Email.")
            if response.status_code == 200:
                print("Acción predicha enviada correctamente al servidor.")
            else:
                print("Error al enviar la acción predicha al servidor. Código de estado:", response.status_code)
        except Exception as e:
            print("Error al enviar la acción predicha al servidor:", str(e))


if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("blue")
    root = customtkinter.CTk()
    root.geometry("1106x706")
    root.resizable(False, False)
    app = AppCamera(root, "SeniorSafe")
