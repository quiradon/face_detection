import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import json
import os
from PIL import Image, ImageTk
import cv2
import threading
import time
from simple_face_recognition_system import SimpleFaceRecognitionSystem


class IntegratedFaceRecognitionApp:
    def __init__(self):
        self.face_system = SimpleFaceRecognitionSystem()
        self.camera = None
        self.running = False
        self.detection_cooldown = 3
        self.last_detections = {}
        self.setup_gui()

    def setup_gui(self):
        """Configura a interface gráfica principal"""
        self.root = tk.Tk()
        self.root.title("Sistema de Reconhecimento Facial - Integrado")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2c3e50")

        # Configurar fechamento da janela
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Frame superior para botão de abrir câmera
        top_frame = tk.Frame(self.root, bg="#2c3e50")
        top_frame.pack(fill="x", padx=10, pady=(10, 0))

        self.open_camera_btn = tk.Button(top_frame, text="Abrir Câmera", command=self.abrir_camera_manual,
                                         font=("Arial", 12, "bold"), bg="#2980b9", fg="white", width=16, height=2)
        self.open_camera_btn.pack(side="left", padx=(0, 10))

        # Criar notebook para abas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Configurar estilo
        style = ttk.Style()
        style.theme_use("clam")

        # Criar as abas
        self.setup_detection_tab()
        self.setup_register_tab()
        self.setup_management_tab()
        self.setup_logs_tab()

    def abrir_camera_manual(self):
        """Seleciona a aba de detecção e inicia a câmera"""
        self.notebook.select(0)  # Seleciona a primeira aba (detecção)
        self.start_camera()

    def setup_detection_tab(self):
        """Configura a aba de detecção em tempo real"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🎥 Detecção em Tempo Real")

        # Frame principal
        main_frame = tk.Frame(detection_frame, bg="#1a252f")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Título
        title_label = tk.Label(main_frame, text="Sistema de Controle de Acesso", 
                              font=("Arial", 18, "bold"), bg="#1a252f", fg="#ecf0f1")
        title_label.pack(pady=(0, 20))

        # Frame do vídeo
        video_frame = tk.Frame(main_frame, bg="#34495e", relief="raised", bd=2)
        video_frame.pack(fill="both", expand=True, pady=(0, 20))
        # Canvas com scroll para a visualização da câmera
        self.video_canvas = tk.Canvas(video_frame, bg="#2c3e50", highlightthickness=0)
        self.video_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        y_scroll = tk.Scrollbar(video_frame, orient="vertical", command=self.video_canvas.yview)
        y_scroll.pack(side="right", fill="y")
        x_scroll = tk.Scrollbar(video_frame, orient="horizontal", command=self.video_canvas.xview)
        x_scroll.pack(side="bottom", fill="x")
        self.video_canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        # Label da imagem dentro do canvas
        self.video_label = tk.Label(self.video_canvas, bg="#2c3e50",
                                   text="Câmera Desconectada", 
                                   font=("Arial", 16), fg="#bdc3c7")
        self.video_label_id = self.video_canvas.create_window((0, 0), window=self.video_label, anchor="nw")
        self.video_label.bind('<Configure>', lambda e: self._update_canvas_scrollregion())

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

        # Frame de configurações
        config_frame = tk.Frame(main_frame, bg="#34495e", relief="raised", bd=2)
        config_frame.pack(fill="x", pady=(0, 20))

        config_title = tk.Label(config_frame, text="Configurações", 
                               font=("Arial", 12, "bold"), bg="#34495e", fg="#ecf0f1")
        config_title.pack(pady=(10, 5))

        # Slider de sensibilidade
        sensitivity_frame = tk.Frame(config_frame, bg="#34495e")
        sensitivity_frame.pack(pady=5)

        tk.Label(sensitivity_frame, text="Sensibilidade:", 
                font=("Arial", 10), bg="#34495e", fg="#bdc3c7").pack(side="left")
        
        self.sensitivity_var = tk.DoubleVar(value=self.face_system.tolerance)
        self.sensitivity_scale = tk.Scale(sensitivity_frame, from_=0.6, to=0.9, 
                                         resolution=0.01, orient="horizontal",
                                         variable=self.sensitivity_var,
                                         command=self.update_sensitivity,
                                         bg="#34495e", fg="#ecf0f1", length=200)
        self.sensitivity_scale.pack(side="left", padx=(10, 0))

        # Frame de informações
        info_frame = tk.Frame(main_frame, bg="#34495e", relief="raised", bd=2)
        info_frame.pack(fill="x")

        # Status de acesso
        self.access_status = tk.Label(info_frame, text="Aguardando detecção...", 
                                     font=("Arial", 16, "bold"), bg="#34495e", fg="#f39c12")
        self.access_status.pack(pady=10)

        # Informações da pessoa detectada
        self.person_info = tk.Label(info_frame, text="", 
                                   font=("Arial", 12), bg="#34495e", fg="#bdc3c7")
        self.person_info.pack(pady=5)

        # Timestamp da última detecção
        self.last_detection_time = tk.Label(info_frame, text="", 
                                           font=("Arial", 10), bg="#34495e", fg="#95a5a6")
        self.last_detection_time.pack(pady=(0, 10))

        # Contador de pessoas cadastradas
        self.update_registered_count()

    def _update_canvas_scrollregion(self):
        self.video_canvas.configure(scrollregion=self.video_canvas.bbox("all"))

    def setup_register_tab(self):
        """Configura a aba de cadastro de novos rostos"""
        register_frame = ttk.Frame(self.notebook)
        self.notebook.add(register_frame, text="➕ Cadastrar Rosto")

        # Título
        title_label = tk.Label(register_frame, text="Cadastro de Novo Rosto", 
                              font=("Arial", 16, "bold"), bg="#34495e", fg="white")
        title_label.pack(fill="x", pady=(0, 20))

        # Frame principal
        main_frame = ttk.Frame(register_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Nome da pessoa
        ttk.Label(main_frame, text="Nome da Pessoa:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))
        self.name_entry = ttk.Entry(main_frame, font=("Arial", 12), width=40)
        self.name_entry.pack(anchor="w", pady=(0, 20))

        # Seleção de imagem
        ttk.Label(main_frame, text="Selecionar Imagem:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(anchor="w", pady=(0, 20))
        
        self.select_image_btn = ttk.Button(image_frame, text="Escolher Imagem", 
                                          command=self.select_image)
        self.select_image_btn.pack(side="left", padx=(0, 10))
        
        # Botão para capturar da câmera
        self.capture_image_btn = ttk.Button(image_frame, text="Capturar da Câmera", command=self.capturar_imagem_camera)
        self.capture_image_btn.pack(side="left", padx=(0, 10))
        
        self.image_path_label = tk.Label(image_frame, text="Nenhuma imagem selecionada", 
                                        fg="gray", font=("Arial", 10))
        self.image_path_label.pack(side="left")

        # Preview da imagem
        self.image_preview_frame = ttk.Frame(main_frame)
        self.image_preview_frame.pack(pady=20)

        # Botão de cadastro
        self.register_btn = ttk.Button(main_frame, text="Cadastrar Rosto", 
                                      command=self.register_face)
        self.register_btn.pack(pady=20)

        # Status
        self.status_label = tk.Label(main_frame, text="Pronto para cadastro", 
                                    fg="green", font=("Arial", 10))
        self.status_label.pack()

        self.selected_image_path = None

    def capturar_imagem_camera(self):
        """Captura o frame atual da câmera para cadastro, isolando e expandindo a região do rosto"""
        if self.camera and self.running:
            ret, frame = self.camera.read()
            if ret:
                # Detectar rosto no frame
                faces = self.face_system.detect_faces(frame)
                if len(faces) == 0:
                    messagebox.showerror("Erro", "Nenhum rosto detectado na imagem da câmera.")
                    return
                # Usar a maior face detectada (mais próxima)
                faces = sorted(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
                bbox = faces[0]['bbox']  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                # Expandir a caixa em 40% para cada lado
                w = x2 - x1
                h = y2 - y1
                expand_x = int(w * 0.4)
                expand_y = int(h * 0.4)
                x1e = max(x1 - expand_x, 0)
                y1e = max(y1 - expand_y, 0)
                x2e = min(x2 + expand_x, frame.shape[1])
                y2e = min(y2 + expand_y, frame.shape[0])
                face_crop = frame[y1e:y2e, x1e:x2e]
                # Salvar apenas a região do rosto expandida
                temp_path = "temp_captura_cadastro.jpg"
                cv2.imwrite(temp_path, face_crop)
                self.selected_image_path = temp_path
                self.image_path_label.config(text="Imagem capturada da câmera", fg="black")
                self.show_image_preview(temp_path)
            else:
                messagebox.showerror("Erro", "Não foi possível capturar imagem da câmera.")
        else:
            messagebox.showwarning("Aviso", "A câmera precisa estar ativa na aba de detecção para capturar.")

    def setup_management_tab(self):
        """Configura a aba de gerenciamento de rostos"""
        management_frame = ttk.Frame(self.notebook)
        self.notebook.add(management_frame, text="👥 Gerenciar Rostos")

        # Título
        title_label = tk.Label(management_frame, text="Gerenciamento de Rostos Cadastrados", 
                              font=("Arial", 16, "bold"), bg="#34495e", fg="white")
        title_label.pack(fill="x", pady=(0, 20))

        # Frame principal
        main_frame = ttk.Frame(management_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Lista de rostos cadastrados
        ttk.Label(main_frame, text="Rostos Cadastrados:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # Scrollbar para a lista
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.faces_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                       font=("Arial", 11), height=15)
        self.faces_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.faces_listbox.yview)

        # Botões de ação
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        self.refresh_btn = ttk.Button(buttons_frame, text="Atualizar Lista", 
                                     command=self.refresh_faces_list)
        self.refresh_btn.pack(side="left", padx=(0, 10))
        
        self.remove_btn = ttk.Button(buttons_frame, text="Remover Selecionado", 
                                    command=self.remove_selected_face)
        self.remove_btn.pack(side="left")

        # Carregar lista inicial
        self.refresh_faces_list()

    def setup_logs_tab(self):
        """Configura a aba de visualização de logs"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs de Acesso")

        # Título
        title_label = tk.Label(logs_frame, text="Histórico de Acessos", 
                              font=("Arial", 16, "bold"), bg="#34495e", fg="white")
        title_label.pack(fill="x", pady=(0, 20))

        # Frame principal
        main_frame = ttk.Frame(logs_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", pady=(0, 20))
        
        self.refresh_logs_btn = ttk.Button(controls_frame, text="Atualizar Logs", 
                                          command=self.refresh_logs)
        self.refresh_logs_btn.pack(side="left", padx=(0, 10))
        
        self.clear_logs_btn = ttk.Button(controls_frame, text="Limpar Logs", 
                                        command=self.clear_logs)
        self.clear_logs_btn.pack(side="left")

        # Área de logs
        self.logs_text = scrolledtext.ScrolledText(main_frame, height=25, font=("Consolas", 10))
        self.logs_text.pack(fill="both", expand=True)

        # Carregar logs inicial
        self.refresh_logs()

    # Métodos da detecção
    def update_sensitivity(self, value):
        """Atualiza a sensibilidade do reconhecimento"""
        self.face_system.tolerance = float(value)

    def start_camera(self):
        """Inicia a câmera e o reconhecimento facial"""
        try:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a câmera")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.camera_status.config(text="Status: Câmera ativa", fg="#27ae60")
            
            self.update_registered_count()
            
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
        
        self.video_label.config(image="", text="Câmera Desconectada")
        self.access_status.config(text="Aguardando detecção...", fg="#f39c12")
        self.person_info.config(text="")
        self.last_detection_time.config(text="")

    def process_video(self):
        """Processa os frames do vídeo em tempo real"""
        frame_skip = 2
        frame_count = 0
        
        while self.running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                if frame_count % frame_skip == 0:
                    results = self.face_system.recognize_faces(frame)
                    self.draw_face_boxes(frame, results)
                    self.process_recognition_results(results)
                
                self.display_frame(frame)
                frame_count += 1
                time.sleep(0.03)
                
            except Exception as e:
                print(f"Erro no processamento de vídeo: {e}")
                break

    def draw_face_boxes(self, frame, results):
        """Desenha retângulos e informações nos rostos detectados"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            
            if result['access_granted']:
                color = (0, 255, 0)
                label = f"{result['name']} - PERMITIDO"
            else:
                color = (0, 0, 255)
                label = f"{result['name']} - NEGADO"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, cv2.FILLED)
            
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
            
            if result['confidence'] > 0:
                confidence_text = f"Conf: {result['confidence']:.2f}"
                cv2.putText(frame, confidence_text, (x1, y2 + 25), font, 0.5, color, 1)

    def process_recognition_results(self, results):
        """Processa os resultados do reconhecimento"""
        current_time = time.time()
        
        for result in results:
            name = result['name']
            access_granted = result['access_granted']
            confidence = result['confidence']
            
            if name in self.last_detections:
                if current_time - self.last_detections[name] < self.detection_cooldown:
                    continue
            
            self.last_detections[name] = current_time
            
            status = "PERMITIDO" if access_granted else "NEGADO"
            self.face_system.log_access(name, status, confidence)
            
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
                person_text += f" (Similaridade: {confidence:.2f})"
        else:
            person_text = "Pessoa não identificada"
        
        self.person_info.config(text=f"{status_msg} - {person_text}")
        self.last_detection_time.config(text=f"Última detecção: {timestamp}")

    def display_frame(self, frame):
        """Exibe o frame na interface Tkinter, com scroll se necessário"""
        try:
            # Tamanho alvo fixo para preview grande
            target_width, target_height = 1280, 960
            frame_resized = cv2.resize(frame, (target_width, target_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
            # Atualizar tamanho do label e canvas
            self.video_label.config(width=target_width, height=target_height)
            self.video_canvas.itemconfig(self.video_label_id, width=target_width, height=target_height)
            self.video_canvas.config(scrollregion=(0, 0, target_width, target_height))
        except Exception as e:
            print(f"Erro ao exibir frame: {e}")

    def update_registered_count(self):
        """Atualiza o contador de pessoas cadastradas"""
        if hasattr(self, 'registered_info'):
            registered_count = len(self.face_system.get_registered_faces())
            self.registered_info.config(text=f"Pessoas cadastradas: {registered_count}")
        else:
            # Criar o label se não existir
            registered_count = len(self.face_system.get_registered_faces())
            # Encontrar o frame de informações na aba de detecção
            detection_tab = self.notebook.nametowidget(self.notebook.tabs()[0])
            info_frames = [w for w in detection_tab.winfo_children() if isinstance(w, tk.Frame)]
            if info_frames:
                main_frame = info_frames[0]
                info_frames_nested = [w for w in main_frame.winfo_children() if isinstance(w, tk.Frame) and w.cget('bg') == '#34495e']
                if info_frames_nested:
                    info_frame = info_frames_nested[-1]  # Último frame com bg #34495e
                    self.registered_info = tk.Label(info_frame, 
                                                   text=f"Pessoas cadastradas: {registered_count}", 
                                                   font=("Arial", 10), bg="#34495e", fg="#3498db")
                    self.registered_info.pack(pady=(0, 10))

    # Métodos do cadastro
    def select_image(self):
        """Abre diálogo para seleção de imagem"""
        file_types = [
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos os arquivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(title="Selecionar Imagem", filetypes=file_types)
        
        if filename:
            self.selected_image_path = filename
            self.image_path_label.config(text=os.path.basename(filename), fg="black")
            self.show_image_preview(filename)

    def show_image_preview(self, image_path):
        """Mostra preview da imagem selecionada"""
        try:
            for widget in self.image_preview_frame.winfo_children():
                widget.destroy()
            
            image = Image.open(image_path)
            image.thumbnail((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            preview_label = tk.Label(self.image_preview_frame, image=photo)
            preview_label.image = photo
            preview_label.pack()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")

    def register_face(self):
        """Cadastra um novo rosto"""
        name = self.name_entry.get().strip()
        
        if not name:
            messagebox.showerror("Erro", "Por favor, digite o nome da pessoa")
            return
        
        if not hasattr(self, 'selected_image_path') or not self.selected_image_path:
            messagebox.showerror("Erro", "Por favor, selecione uma imagem")
            return
        
        if name in self.face_system.get_registered_faces():
            if not messagebox.askyesno("Confirmar", f"Já existe um cadastro para '{name}'. Deseja substituir?"):
                return
            self.face_system.remove_face(name)
        
        self.status_label.config(text="Processando...", fg="orange")
        self.root.update()
        
        try:
            success, message = self.face_system.add_new_face(self.selected_image_path, name)
            
            if success:
                self.status_label.config(text=message, fg="green")
                self.name_entry.delete(0, tk.END)
                self.selected_image_path = None
                self.image_path_label.config(text="Nenhuma imagem selecionada", fg="gray")
                
                for widget in self.image_preview_frame.winfo_children():
                    widget.destroy()
                
                self.refresh_faces_list()
                self.update_registered_count()
                messagebox.showinfo("Sucesso", message)
            else:
                self.status_label.config(text=message, fg="red")
                messagebox.showerror("Erro", message)
                
        except Exception as e:
            error_msg = f"Erro inesperado: {str(e)}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Erro", error_msg)

    # Métodos do gerenciamento
    def refresh_faces_list(self):
        """Atualiza a lista de rostos cadastrados"""
        self.faces_listbox.delete(0, tk.END)
        faces = self.face_system.get_registered_faces()
        
        for face in sorted(faces):
            self.faces_listbox.insert(tk.END, face)

    def remove_selected_face(self):
        """Remove o rosto selecionado"""
        selection = self.faces_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("Aviso", "Por favor, selecione um rosto para remover")
            return
        
        name = self.faces_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirmar", f"Deseja realmente remover o cadastro de '{name}'?"):
            success, message = self.face_system.remove_face(name)
            
            if success:
                messagebox.showinfo("Sucesso", message)
                self.refresh_faces_list()
                self.update_registered_count()
            else:
                messagebox.showerror("Erro", message)

    # Métodos dos logs
    def refresh_logs(self):
        """Atualiza os logs de acesso"""
        self.logs_text.delete(1.0, tk.END)
        
        logs = self.face_system.get_logs()
        
        if not logs:
            self.logs_text.insert(tk.END, "Nenhum log de acesso encontrado.\n")
            return
        
        logs_sorted = sorted(logs, key=lambda x: x['timestamp'], reverse=True)
        
        for log in logs_sorted:
            timestamp = datetime.fromisoformat(log['timestamp']).strftime("%d/%m/%Y %H:%M:%S")
            
            log_line = f"[{timestamp}] {log['name']} - {log['status']}"
            if log.get('confidence'):
                log_line += f" (Confiança: {log['confidence']:.2%})"
            log_line += "\n"
            
            self.logs_text.insert(tk.END, log_line)

    def clear_logs(self):
        """Limpa todos os logs"""
        if messagebox.askyesno("Confirmar", "Deseja realmente limpar todos os logs?"):
            try:
                if os.path.exists(self.face_system.logs_file):
                    os.remove(self.face_system.logs_file)
                self.refresh_logs()
                messagebox.showinfo("Sucesso", "Logs limpos com sucesso")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao limpar logs: {str(e)}")

    def on_closing(self):
        """Função chamada ao fechar a janela"""
        self.stop_camera()
        self.root.destroy()

    def run(self):
        """Executa a aplicação"""
        self.root.mainloop()


if __name__ == "__main__":
    app = IntegratedFaceRecognitionApp()
    app.run()
