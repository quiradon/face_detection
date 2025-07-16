import cv2
import json
import datetime
import os

def preprocessar_face(face_img):
    """
    Aplica o mesmo pré-processamento usado no treinamento
    Esta função garante consistência entre treinamento e detecção
    """
    # 1. Filtro bilateral para reduzir ruído preservando bordas
    face_preprocessada = cv2.bilateralFilter(face_img, 9, 75, 75)
    
    # 2. CLAHE para melhorar o contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_preprocessada = clahe.apply(face_preprocessada)
    
    # 3. Equalização de histograma
    face_preprocessada = cv2.equalizeHist(face_preprocessada)
    
    # 4. Normalização do contraste
    face_preprocessada = cv2.normalize(face_preprocessada, None, 0, 255, cv2.NORM_MINMAX)
    
    return face_preprocessada

def obter_nome_camera(camera_id):
    """Tenta obter o nome da câmera usando diferentes métodos"""
    try:
        # Método 1: Usar propriedade do OpenCV (nem sempre funciona)
        cap = cv2.VideoCapture(camera_id)
        backend_name = cap.getBackendName()
        cap.release()
        
        # Método 2: Usar wmi no Windows para obter informações mais detalhadas
        try:
            import wmi
            c = wmi.WMI()
            cameras = c.Win32_PnPEntity(ConfigManagerErrorCode=0)
            
            # Procura por dispositivos de câmera
            camera_keywords = ['camera', 'webcam', 'imaging', 'video']
            device_names = []
            
            for device in cameras:
                if device.Name and any(keyword.lower() in device.Name.lower() for keyword in camera_keywords):
                    device_names.append(device.Name)
            
            # Retorna o nome do dispositivo baseado no índice
            if camera_id < len(device_names):
                return device_names[camera_id]
            
        except ImportError:
            # wmi não está disponível, tenta método alternativo
            try:
                import subprocess
                result = subprocess.run(['powershell', '-Command', 
                    'Get-PnpDevice | Where-Object {$_.Class -eq "Camera" -or $_.FriendlyName -like "*camera*" -or $_.FriendlyName -like "*webcam*"} | Select-Object -ExpandProperty FriendlyName'], 
                    capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    device_names = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    if camera_id < len(device_names):
                        return device_names[camera_id]
            except Exception:
                pass
        except Exception:
            pass
        
        # Método 3: Nomes padrão baseados no backend
        if 'DSHOW' in backend_name:
            return f"DirectShow Camera {camera_id}"
        elif 'MSMF' in backend_name:
            return f"Media Foundation Camera {camera_id}"
        else:
            return f"Camera {camera_id} ({backend_name})"
            
    except Exception:
        return f"Camera {camera_id}"

def listar_cameras():
    """Lista todas as câmeras disponíveis no sistema com seus nomes"""
    print("🔍 Procurando câmeras disponíveis...")
    cameras_disponiveis = []
    cameras_info = {}
    
    # Testa até 10 câmeras (normalmente é suficiente)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        # Configura um timeout menor para acelerar a detecção
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.read()[0]:
            cameras_disponiveis.append(i)
            
            # Obtém informações da câmera
            largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            nome = obter_nome_camera(i)
            
            cameras_info[i] = {
                'nome': nome,
                'resolucao': f"{largura}x{altura}"
            }
            
            print(f"📷 Câmera {i}: {nome} ({largura}x{altura})")
        cap.release()
    
    return cameras_disponiveis, cameras_info

def selecionar_camera():
    """Permite ao usuário selecionar uma câmera da lista de câmeras disponíveis"""
    cameras_disponiveis, cameras_info = listar_cameras()
    
    if not cameras_disponiveis:
        print("❌ Nenhuma câmera encontrada no sistema!")
        return None
    
    print(f"\n� Resumo das câmeras disponíveis:")
    for cam_id in cameras_disponiveis:
        info = cameras_info[cam_id]
        print(f"   {cam_id}: {info['nome']} - {info['resolucao']}")
    
    while True:
        try:
            escolha = input(f"\n🎯 Digite o número da câmera que deseja usar (ou 'q' para sair): ")
            
            if escolha.lower() == 'q':
                print("👋 Saindo...")
                return None
            
            camera_id = int(escolha)
            
            if camera_id in cameras_disponiveis:
                # Testa se a câmera ainda está funcionando
                info = cameras_info[camera_id]
                print(f"🔍 Testando {info['nome']}...")
                cap = cv2.VideoCapture(camera_id)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    print(f"✅ {info['nome']} selecionada com sucesso!")
                    return camera_id
                else:
                    print(f"❌ Erro: {info['nome']} não está respondendo. Tente outra.")
            else:
                print(f"❌ Câmera {camera_id} não está disponível. Escolha uma das opções: {cameras_disponiveis}")
                
        except ValueError:
            print("❌ Por favor, digite um número válido ou 'q' para sair.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operação cancelada pelo usuário.")
            return None

def registrar_log(nome, acesso_permitido):
    # Cria o diretório de logs se não existir
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Nome do arquivo de log (um arquivo por dia)
    data_atual = datetime.datetime.now()
    arquivo_log = f"logs/log_{data_atual.strftime('%Y-%m-%d')}.txt"
    
    # Prepara a mensagem de log
    timestamp = data_atual.strftime("%Y-%m-%d %H:%M:%S")
    status = "PERMITIDO" if acesso_permitido else "NEGADO"
    mensagem = f"[{timestamp}] Acesso {status} - Pessoa: {nome}\n"
    
    # Registra no arquivo de log
    with open(arquivo_log, 'a', encoding='utf-8') as f:
        f.write(mensagem)

def iniciar_reconhecimento(camera_id=0):
    # Carrega o classificador para detecção facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Carrega o modelo treinado
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read('classificador.yml')
    
    # Carrega o dicionário de nomes
    with open('nomes.json', 'r') as f:
        nomes = json.load(f)
    
    # Inicia a captura de vídeo com a câmera selecionada
    cap = cv2.VideoCapture(camera_id)
    
    # Verifica se a câmera foi aberta corretamente
    if not cap.isOpened():
        print(f"❌ Erro: Não foi possível abrir a câmera {camera_id}")
        return False
    
    # Obtém informações da câmera
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 Câmera {camera_id} configurada: {largura}x{altura} @ {fps}fps")
    
    # Configura a fonte para o texto na tela
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    print("🎯 Reconhecimento facial ativo!")
    print("💡 Pressione 'q' para sair\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem da câmera")
            break
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta faces na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Expande a região do rosto em 10% para cada lado
            expand_x = int(w * 0.1)
            expand_y = int(h * 0.1)
            
            # Calcula novas coordenadas com a expansão, garantindo que não ultrapassem os limites da imagem
            new_x = max(0, x - expand_x)
            new_y = max(0, y - expand_y)
            new_w = min(frame.shape[1] - new_x, w + 2 * expand_x)
            new_h = min(frame.shape[0] - new_y, h + 2 * expand_y)
            
            # Extrai a região da face expandida
            face_roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]
            
            try:
                # Aplica o MESMO pré-processamento usado no treinamento
                face_roi = preprocessar_face(face_roi)
                
                # Tenta reconhecer a face
                id_previsto, confianca = reconhecedor.predict(face_roi)
                
                # Define um limiar de confiança (quanto menor, mais preciso é o reconhecimento)
                nome = nomes.get(str(id_previsto), "Desconhecido")
                
                # Define os níveis de confiança (valores ajustados para escala real do LBPH)
                if confianca > 1000:  # Provavelmente um erro no reconhecimento
                    nome = "Desconhecido"
                    cor = (0, 0, 255)  # Vermelho
                    status = "ERRO - Reconhecimento falhou"
                    registrar_log(nome, False)
                elif confianca < 50:  # Reconhecimento bom
                    cor = (0, 255, 128)  # Verde claro
                    status = "PERMITIDO (Fiel)"
                    registrar_log(nome, True)
                elif confianca < 70:  # Reconhecimento aceitável
                    cor = (0, 255, 0)  # Amarelo
                    status = "PERMITIDO (Verificar)"
                    registrar_log(nome, True)
                else:  # Reconhecimento duvidoso
                    cor = (0, 0, 255)  # Vermelho
                    status = "NEGADO"
                    registrar_log(nome, False)
                
                # Desenha o retângulo e textos usando as coordenadas expandidas
                cv2.rectangle(frame, (new_x, new_y), (new_x+new_w, new_y+new_h), cor, 2)
                
                # Mostra o nome e status
                texto_status = f"Status: {status}"
                cv2.putText(frame, texto_status, (new_x, new_y-10), fonte, 0.5, cor, 2)
                
                # Mostra o nome detectado
                texto_nome = f"Nome: {nome}"
                cv2.putText(frame, texto_nome, (new_x, new_y-25), fonte, 0.5, cor, 2)
                
                # Mostra a pontuação de confiança
                texto_confianca = f"Confianca: {confianca:.1f}"
                cv2.putText(frame, texto_confianca, (new_x, new_y-40), fonte, 0.5, cor, 2)
                
            except Exception as e:
                print(f"Erro no reconhecimento: {str(e)}")
        
        # Mostra o frame
        cv2.imshow('Reconhecimento Facial', frame)
        
        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    try:
        print("🚀 === Sistema de Reconhecimento Facial ===")
        print("📋 Iniciando configuração de câmera...\n")
        
        # Seleciona a câmera
        camera_id = selecionar_camera()
        
        if camera_id is not None:
            print(f"\n🎬 Iniciando reconhecimento com câmera {camera_id}...")
            print("💡 Pressione 'q' para sair do reconhecimento")
            print("⚡ Carregando modelos...\n")
            
            sucesso = iniciar_reconhecimento(camera_id)
            
            if not sucesso:
                print("❌ Falha ao iniciar o reconhecimento facial.")
            else:
                print("✅ Reconhecimento facial encerrado com sucesso.")
        else:
            print("⚠️  Operação cancelada.")
            
    except Exception as e:
        print(f"❌ Erro ao iniciar o reconhecimento: {str(e)}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrompido pelo usuário.")
    
    print("\n👋 Sistema encerrado.")
