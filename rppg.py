import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from picamera2 import Picamera2
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

# Funciones de procesamiento de señales

def bandpass_filter(signal, fs, lowcut=0.75, highcut=3.0, order=5):
    # Filtro de paso banda para eliminar el ruido de la señal
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal

# Calcula la frecuencia cardiaca a partir de las señales de las regiones de interés

def calculate_heart_rate(roi_signals, fps=30):
    # Normalizar las señales (restar la media y dividir por la desviación estándar)
    roi_signals = (roi_signals - np.mean(roi_signals, axis=0)) / np.std(roi_signals, axis=0)
    if np.isnan(roi_signals).any() or np.isinf(roi_signals).any():
        return None
    
    # Determinar el número de componentes para la descomposición ICA
    n_components = min(roi_signals.shape[1], roi_signals.shape[0] - 1)
    if n_components < 1:
        return None
    
    # Aplicar FastICA para separar las fuentes independientes
    ica = FastICA(n_components=n_components)
    try:
        ica_signals = ica.fit_transform(roi_signals)
    except ValueError:
        return None
    
    # Aplicar el filtro de paso banda a las componentes independientes
    filtered_components = bandpass_filter(ica_signals, fps)
    
    # Inicializar los valores máximos para encontrar la frecuencia cardiaca
    max_peak = -np.inf
    heart_rate = 0
    
    # Analizar cada componente y calcular la FFT para encontrar el pico dominante
    for component in filtered_components.T:
        fft_values = np.abs(np.fft.rfft(component))
        freqs = np.fft.rfftfreq(len(component), d=1/fps)
        idx_band = np.where((freqs >= 0.75) & (freqs <= 3.0))
        if len(idx_band[0]) == 0:
            continue
        
        # Obtener la frecuencia de pico más alta dentro del rango de interés
        fft_band = fft_values[idx_band]
        freqs_band = freqs[idx_band]
        peak_idx = np.argmax(fft_band)
        peak_freq = freqs_band[peak_idx]
        peak_amplitude = fft_band[peak_idx]
        
        if peak_amplitude > max_peak:
            max_peak = peak_amplitude
            heart_rate = peak_freq * 60  # Convertir a BPM
    
    if heart_rate == 0:
        return None
    else:
        return heart_rate

# Dibuja los límites alrededor de la cara detectada

def draw_boundary(frame, classifier, scale, minNeighbors, color, text):
    # Convertir la imagen a escala de grises para la detección de la cara
    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    features = classifier.detectMultiScale(gray_img, scale, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
        break  # Solo procesar la primera cara detectada
    return coords, frame

# Calcula las posiciones de los marcadores en la cara

def calculate_face_markers(coords):
    if len(coords) == 4:
        x, y, w, h = coords
        # Definir las posiciones de los marcadores (frente y mejillas)
        forehead_pos = (x + w // 2 - 30, y + 5, x + w // 2 + 30, y + 35)
        cheek_left_pos = (x + int(w * 0.25) - 12, y + int(h * 0.6), x + int(w * 0.25) + 12, y + int(h * 0.6) + 25)
        cheek_right_pos = (x + int(w * 0.75) - 12, y + int(h * 0.6), x + int(w * 0.75) + 12, y + int(h * 0.6) + 25)
        return forehead_pos, cheek_left_pos, cheek_right_pos
    return None, None, None

# Dibuja los marcadores en las regiones de interés de la cara

def draw_face_markers(frame, forehead_pos, cheek_left_pos, cheek_right_pos):
    pink = (255, 0, 255)  # Color rosa para los marcadores
    if forehead_pos and cheek_left_pos and cheek_right_pos:
        # Dibujar rectángulos para las regiones de interés
        cv2.rectangle(frame, forehead_pos[:2], forehead_pos[2:], pink, 2)
        cv2.rectangle(frame, cheek_left_pos[:2], cheek_left_pos[2:], pink, 2)
        cv2.rectangle(frame, cheek_right_pos[:2], cheek_right_pos[2:], pink, 2)
    return frame

# Detecta la cara y calcula las regiones de interés

def detect(frame, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    # Dibuja los límites alrededor de la cara detectada
    coords, frame = draw_boundary(frame, faceCascade, 1.1, 10, color['green'], "Cara")
    # Calcula las posiciones de las regiones de interés (frente y mejillas)
    forehead_pos, cheek_left_pos, cheek_right_pos = calculate_face_markers(coords)
    # Dibuja los marcadores en las regiones de interés
    frame = draw_face_markers(frame, forehead_pos, cheek_left_pos, cheek_right_pos)
    return frame, coords, forehead_pos, cheek_left_pos, cheek_right_pos

# Crea la composición de los valores de las ROIs

def create_composition(frame, forehead_pos, cheek_left_pos, cheek_right_pos):
    if forehead_pos and cheek_left_pos and cheek_right_pos:
        # Extraer los valores de los píxeles de las regiones de interés
        forehead = frame[forehead_pos[1]:forehead_pos[3], forehead_pos[0]:forehead_pos[2]]
        cheek_left = frame[cheek_left_pos[1]:cheek_left_pos[3], cheek_left_pos[0]:cheek_left_pos[2]]
        cheek_right = frame[cheek_right_pos[1]:cheek_right_pos[3], cheek_right_pos[0]:cheek_right_pos[2]]
        if forehead.size and cheek_left.size and cheek_right.size:
            # Calcular la media de los valores de los píxeles en cada región de interés
            forehead_mean = np.mean(forehead.reshape(-1, 3), axis=0)
            cheek_left_mean = np.mean(cheek_left.reshape(-1, 3), axis=0)
            cheek_right_mean = np.mean(cheek_right.reshape(-1, 3), axis=0)
            # Concatenar los valores medios para formar una composición
            composition = np.concatenate((forehead_mean, cheek_left_mean, cheek_right_mean))
            return composition
    return None

# --- Inicio del código de la interfaz gráfica ---

class HeartRateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Rate Monitor")
        self.root.configure(bg="white")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.camera_display_on = False
        self.picam2 = None
        self.frame = None
        self.roi_signals_buffer = []
        self.fps = 30  # Ajusta según tu cámara
        self.heart_rates = []
        self.avg_heart_rates_buffer = []
        self.frame_count = 0

        # Configuración de la interfaz gráfica
        self.setup_gui()

        # Inicializar la cámara
        self.initialize_camera()

    def setup_gui(self):
        # Configurar la cuadrícula de la ventana principal
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        # Visualización de video
        self.video_frame = tk.Frame(self.root, bg="white")
        self.video_frame.grid(row=0, column=0, rowspan=2, columnspan=2, sticky="nsew")
        self.video_label = tk.Label(self.video_frame, text="Monitor Apagado", bg="black", fg="white", font=("Helvetica", 20))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Botones de control
        self.control_frame = tk.Frame(self.root, bg="white")
        self.control_frame.grid(row=2, column=0, rowspan=2, columnspan=2, sticky="nsew")

        # Cargar iconos para los botones y ajustar tamaño
        self.start_icon = ImageTk.PhotoImage(Image.open("start_icon.png").resize((30, 30), Image.ANTIALIAS))
        self.stop_icon = ImageTk.PhotoImage(Image.open("stop_icon.png").resize((30, 30), Image.ANTIALIAS))
        self.reset_icon = ImageTk.PhotoImage(Image.open("reset_icon.png").resize((30, 30), Image.ANTIALIAS))

        self.start_button = tk.Button(self.control_frame, text="Mostrar Cámara", image=self.start_icon, compound="left", command=self.start_camera, font=("Helvetica", 14), bg="white")
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.control_frame, text="Ocultar Cámara", image=self.stop_icon, compound="left", command=self.stop_camera, state=tk.DISABLED, font=("Helvetica", 14), bg="white")
        self.stop_button.pack(pady=10)

        self.reset_button = tk.Button(self.control_frame, text="Reiniciar", image=self.reset_icon, compound="left", command=self.reset, font=("Helvetica", 14), bg="white")
        self.reset_button.pack(pady=10)

        # Gráfica de frecuencia cardíaca
        self.graph_frame = tk.Frame(self.root, bg="white")
        self.graph_frame.grid(row=0, column=2, rowspan=3, columnspan=2, sticky="nsew")
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("Frecuencia Cardíaca en Tiempo Real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Frecuencia Cardíaca (BPM)")
        self.ax.grid(True)
        self.line_real, = self.ax.plot([], [], 'g-', linewidth=2, label="Frecuencia Real")
        self.line_avg, = self.ax.plot([], [], 'b-', linewidth=2, label="Frecuencia Media")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Etiquetas para mostrar la frecuencia cardíaca en tiempo real y la media global
        self.values_frame = tk.Frame(self.root, bg="white")
        self.values_frame.grid(row=3, column=2, rowspan=1, columnspan=2, sticky="nsew")
        self.heart_rate_label = tk.Label(self.values_frame, text="Frecuencia Cardíaca: --- BPM", font=("Helvetica", 20), fg="green", bg="white")
        self.heart_rate_label.pack(pady=10)
        self.global_avg_label = tk.Label(self.values_frame, text="Media Global: --- BPM", font=("Helvetica", 16), fg="blue", bg="white")
        self.global_avg_label.pack(pady=10)

    def initialize_camera(self):
        try:
            # Inicializar la cámara Picamera2
            self.picam2 = Picamera2()
            time.sleep(2)  # Añadir un retraso para permitir que la cámara se inicie correctamente
            preview_config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            self.picam2.configure(preview_config)
            self.picam2.start()
        except RuntimeError:
            messagebox.showerror("Error", "No se pudo acceder a la cámara. Por favor, asegúrese de que la cámara está conectada correctamente.")

    def start_camera(self):
        if not self.camera_display_on:
            self.camera_display_on = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_frame()

    def stop_camera(self):
        if self.camera_display_on:
            self.camera_display_on = False
            self.video_label.config(image='', text="Monitor Apagado", bg="black", fg="white")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def reset(self):
        # Guardar los datos actuales en un archivo CSV si existen valores en los buffers
        if self.heart_rates or self.avg_heart_rates_buffer:
            file_index = 1
            file_name = 'heart_rate_data.csv'
            while os.path.exists(os.path.join(os.getcwd(), file_name)):
                file_name = f'heart_rate_data_{file_index}.csv'
                file_index += 1

            # Guardar los datos de frecuencia cardíaca en un archivo CSV
            df = pd.DataFrame({
                'Frecuencia Real (BPM)': self.heart_rates,
                'Frecuencia Media (BPM)': self.avg_heart_rates_buffer
            })
            df.to_csv(os.path.join(os.getcwd(), file_name), index=False)

        # Reiniciar los valores
        self.roi_signals_buffer = []
        self.heart_rates = []
        self.avg_heart_rates_buffer = []
        self.ax.clear()
        self.ax.set_title("Frecuencia Cardíaca en Tiempo Real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Frecuencia Cardíaca (BPM)")
        self.ax.grid(True)
        self.canvas.draw()
        self.heart_rate_label.config(text="Frecuencia Cardíaca: --- BPM")
        self.global_avg_label.config(text="Media Global: --- BPM")
        if self.camera_display_on:
            self.update_frame()

    def update_frame(self):
        if self.camera_display_on and self.picam2:
            self.frame_count += 1
            if self.frame_count % 5 == 0:  # Procesar cada 5 frames para optimizar la eficiencia
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_copy = frame.copy()

                # Detección de rostro y ROIs
                frame_copy, coords, forehead_pos, cheek_left_pos, cheek_right_pos = detect(frame_copy, faceCascade)

                # Procesamiento de las ROIs y cálculo de la frecuencia cardíaca
                if coords and len(coords) == 4:
                    composition = create_composition(frame, forehead_pos, cheek_left_pos, cheek_right_pos)
                    if composition is not None:
                        self.roi_signals_buffer.append(composition)
                        if len(self.roi_signals_buffer) > 150:
                            self.roi_signals_buffer.pop(0)
                        if len(self.roi_signals_buffer) == 150:
                            roi_signals_array = np.array(self.roi_signals_buffer)
                            heart_rate = calculate_heart_rate(roi_signals_array, fps=self.fps)
                            if heart_rate:
                                # Añadir el valor de la frecuencia cardíaca en tiempo real a la lista (máximo de 100 valores)
                                if len(self.heart_rates) >= 100:
                                    self.heart_rates.pop(0)
                                self.heart_rates.append(heart_rate)

                                # Calcular la media de la lista de frecuencias cardíacas (máximo de 100 valores)
                                avg_heart_rate = np.mean(self.heart_rates)
                                if len(self.avg_heart_rates_buffer) >= 100:
                                    self.avg_heart_rates_buffer.pop(0)
                                self.avg_heart_rates_buffer.append(avg_heart_rate)
                                
                                self.update_graph()

                                # Mostrar la frecuencia cardíaca en tiempo real
                                self.heart_rate_label.config(text=f"Frecuencia Cardíaca: {int(self.heart_rates[-1])} BPM")

                                # Actualizar la media global de frecuencia cardíaca
                                self.global_avg_label.config(text=f"Media Global: {int(self.avg_heart_rates_buffer[-1])} BPM")
                            else:
                                print('No se pudo estimar la frecuencia cardíaca.')

                            # Mantener el búfer para el cálculo continuo
                            self.roi_signals_buffer.pop(0)
                            
                # Conversión del frame para mostrar en Tkinter
                img = Image.fromarray(frame_copy)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text='')

            # Programar la siguiente actualización de frame
            self.root.after(10, self.update_frame)

    def update_graph(self):
        x_data = np.arange(len(self.heart_rates))[-33:]
        y_data = [int(hr) for hr in self.heart_rates[-33:]]
        avg_data = [int(hr) for hr in self.avg_heart_rates_buffer[-33:]]
        self.ax.clear()
        self.ax.set_title("Frecuencia Cardíaca en Tiempo Real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Frecuencia Cardíaca (BPM)")
        self.ax.grid(True)
        self.ax.plot(x_data, y_data, 'g-', linewidth=2, label="Frecuencia Real")
        self.ax.plot(x_data, avg_data, 'b-', linewidth=2, label="Frecuencia Media")
        self.ax.legend()
        self.canvas.draw()

    def on_closing(self):
        if self.picam2:
            self.picam2.stop()
        self.root.quit()
        self.root.destroy()

# Inicializar el clasificador Haar Cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Inicializar la ventana de Tkinter
root = tk.Tk()
app = HeartRateApp(root)
root.mainloop()
