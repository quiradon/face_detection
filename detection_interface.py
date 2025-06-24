import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import threading
import time
from face_recognition_system import FaceRecognitionSystem


class DetectionInterface:
    def __init__(self):
        self.face_system = FaceRecognitionSystem()
        self.camera = None
        self.running = False
        self.setup_gui()
        
        # Configurações de detecção
        self.detection_cooldown = 3  # segundos entre detecções da mesma pessoa
        self.last_detections = {}  # armazena última detecção por pessoa

    def setup_gui(self):
        """Configura a interface gráfica de detecção"""
        self.root = tk.Tk()
        self.root.title("Sistema de Reconhecimento Facial - Detecção")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a252f")
        
        # Configurar fechamento da janela
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Frame principal
        main_frame = tk.Frame(self.root, bg="#1a252f")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Sistema de Controle de Acesso", 
                              font=("Arial", 20, "bold"), bg="#1a252f", fg="#ecf0f1")
        title_label.pack(pady=(0, 20))

        # Frame do vídeo
        video_frame = tk.Frame(main_frame, bg="#34495e", relief="raised", bd=2)
        video_frame.pack(pady=(0, 20))

        # Label para o vídeo
        self.video_label = tk.Label(video_frame, bg="#2c3e50", width=80, height=24,
                                   text="Câmera Desconectada", font=("Arial", 16),
                                   fg="#bdc3c7")
        self.video_label.pack(padx=10, pady=10)

        # Frame de controles
        controls_frame = tk.Frame(main_frame, bg="#1a252f")
        controls_frame.pack(fill="x", pady=(0, 20))

        # Botões de controle
        self.start_btn = tk.Button(controls_frame, text="Iniciar Câmera", 
                                  command=self.start_camera, font=("Arial", 12, "bold"),
                                  bg="#27ae60", fg="white", width=15, height=2)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(controls_frame, text="Parar Câmera", 
                                 command=self.stop_camera, font=("Arial", 12, "bold"),
                                 bg="#e74c3c", fg="white", width=15, height=2, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))

        # Status da câmera
        self.camera_status = tk.Label(controls_frame, text="Status: Desconectada", 
                                     font=("Arial", 12), bg="#1a252f", fg="#e74c3c")
        self.camera_status.pack(side="left", padx=(20, 0))

        # Frame de informações
        info_frame = tk.Frame(main_frame, bg="#34495e", relief="raised", bd=2)
        info_frame.pack(fill="x")

        # Título das informações
        info_title = tk.Label(info_frame, text="Informações de Acesso", 
                             font=("Arial", 14, "bold"), bg="#34495e", fg="#ecf0f1")
        info_title.pack(pady=(10, 5))

        # Status de acesso
        self.access_status = tk.Label(info_frame, text="Aguardando detecção...", 
                                     font=("Arial", 16, "bold"), bg="#34495e", fg="#f39c12")
        self.access_status.pack(pady=5)

        # Informações da pessoa detectada
        self.person_info = tk.Label(info_frame, text="", 
                                   font=("Arial", 12), bg="#34495e", fg="#bdc3c7")
        self.person_info.pack(pady=5)

        # Timestamp da última detecção
        self.last_detection_time = tk.Label(info_frame, text="", 
                                           font=("Arial", 10), bg="#34495e", fg="#95a5a6")
        self.last_detection_time.pack(pady=(0, 10))

        # Contador de pessoas cadastradas
        registered_count = len(self.face_system.get_registered_faces())
        self.registered_info = tk.Label(info_frame, 
                                       text=f"Pessoas cadastradas: {registered_count}", 
                                       font=("Arial", 10), bg="#34495e", fg="#3498db")
        self.registered_info.pack(pady=(0, 10))

    def start_camera(self):
        """Inicia a câmera e o reconhecimento facial"""
        try:
            # Tentar conectar à câmera
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a câmera")
                return
            
            # Configurar resolução da câmera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.camera_status.config(text="Status: Conectada", fg="#27ae60")
            
            # Iniciar thread de processamento de vídeo
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")

    def stop_camera(self):
        """Para a câmera e o reconhecimento facial"""
        self.running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.camera_status.config(text="Status: Desconectada", fg="#e74c3c")
        
        # Limpar o label do vídeo
        self.video_label.config(image="", text="Câmera Desconectada")
        self.access_status.config(text="Aguardando detecção...", fg="#f39c12")
        self.person_info.config(text="")
        self.last_detection_time.config(text="")

    def process_video(self):
        """Processa os frames do vídeo em tempo real"""
        while self.running and self.camera:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    break
                
                # Espelhar o frame horizontalmente
                frame = cv2.flip(frame, 1)
                
                # Realizar reconhecimento facial
                results, face_locations = self.face_system.recognize_face(frame)
                
                # Desenhar retângulos e labels nos rostos detectados
                self.draw_face_boxes(frame, results, face_locations)
                
                # Processar resultados de reconhecimento
                self.process_recognition_results(results)
                
                # Converter frame para exibição no Tkinter
                self.display_frame(frame)
                
                # Pequena pausa para não sobrecarregar o processamento
                time.sleep(0.03)
                
            except Exception as e:
                print(f"Erro no processamento de vídeo: {e}")
                break

    def draw_face_boxes(self, frame, results, face_locations):
        """Desenha retângulos e informações nos rostos detectados"""
        for (top, right, bottom, left), result in zip(face_locations, results):
            # Escalar coordenadas de volta ao tamanho original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Definir cor baseada no resultado
            if result['access_granted']:
                color = (0, 255, 0)  # Verde para acesso permitido
                label = f"{result['name']} - PERMITIDO"
            else:
                color = (0, 0, 255)  # Vermelho para acesso negado
                label = f"{result['name']} - NEGADO"
            
            # Desenhar retângulo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Desenhar fundo para o texto
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Desenhar texto
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Mostrar confiança se disponível
            if result['confidence'] > 0:
                confidence_text = f"Conf: {result['confidence']:.1%}"
                cv2.putText(frame, confidence_text, (left + 6, top - 6), font, 0.5, color, 1)

    def process_recognition_results(self, results):
        """Processa os resultados do reconhecimento e atualiza a interface"""
        current_time = time.time()
        
        for result in results:
            name = result['name']
            access_granted = result['access_granted']
            confidence = result['confidence']
            
            # Verificar cooldown para evitar logs duplicados
            if name in self.last_detections:
                if current_time - self.last_detections[name] < self.detection_cooldown:
                    continue
            
            # Atualizar timestamp da última detecção
            self.last_detections[name] = current_time
            
            # Registrar no log
            status = "PERMITIDO" if access_granted else "NEGADO"
            self.face_system.log_access(name, status, confidence)
            
            # Atualizar interface
            self.update_access_info(name, access_granted, confidence)

    def update_access_info(self, name, access_granted, confidence):
        """Atualiza as informações de acesso na interface"""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        if access_granted:
            self.access_status.config(text="ACESSO PERMITIDO", fg="#27ae60")
            status_msg = "Acesso liberado"
        else:
            self.access_status.config(text="ACESSO NEGADO", fg="#e74c3c")
            status_msg = "Acesso bloqueado"
        
        if name != "Desconhecido":
            person_text = f"Pessoa: {name}"
            if confidence > 0:
                person_text += f" (Confiança: {confidence:.1%})"
        else:
            person_text = "Pessoa não identificada"
        
        self.person_info.config(text=f"{status_msg} - {person_text}")
        self.last_detection_time.config(text=f"Última detecção: {timestamp}")

    def display_frame(self, frame):
        """Exibe o frame na interface Tkinter"""
        try:
            # Redimensionar frame para a interface
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Converter para formato PIL
            from PIL import Image, ImageTk
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Atualizar label do vídeo
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Manter referência
            
        except Exception as e:
            print(f"Erro ao exibir frame: {e}")

    def on_closing(self):
        """Função chamada ao fechar a janela"""
        self.stop_camera()
        self.root.destroy()

    def run(self):
        """Executa a interface"""
        self.root.mainloop()


if __name__ == "__main__":
    app = DetectionInterface()
    app.run()
