import cv2
import json
import datetime
import os

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

def iniciar_reconhecimento():
    # Carrega o classificador para detecção facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Carrega o modelo treinado
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read('classificador.yml')
    
    # Carrega o dicionário de nomes
    with open('nomes.json', 'r') as f:
        nomes = json.load(f)
    
    # Inicia a captura de vídeo
    cap = cv2.VideoCapture(0)
    
    # Configura a fonte para o texto na tela
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    print("Pressione 'q' para sair")
    
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
            # Extrai a região da face
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                # Aplica o mesmo pré-processamento usado no treinamento
                face_roi = cv2.equalizeHist(face_roi)
                face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
                face_roi = cv2.normalize(face_roi, None, 0, 255, cv2.NORM_MINMAX)
                
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
                elif confianca < 30:  # Reconhecimento muito confiável
                    cor = (0, 255, 0)  # Verde
                    status = "PERMITIDO (Alta Confiança)"
                    registrar_log(nome, True)
                elif confianca < 50:  # Reconhecimento bom
                    cor = (0, 255, 128)  # Verde claro
                    status = "PERMITIDO"
                    registrar_log(nome, True)
                elif confianca < 70:  # Reconhecimento aceitável
                    cor = (0, 255, 255)  # Amarelo
                    status = "PERMITIDO (Verificar)"
                    registrar_log(nome, True)
                else:  # Reconhecimento duvidoso
                    cor = (0, 0, 255)  # Vermelho
                    status = "NEGADO"
                    registrar_log(nome, False)
                
                # Desenha o retângulo e textos
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
                
                # Mostra o nome e status
                texto_status = f"Status: {status}"
                cv2.putText(frame, texto_status, (x, y-10), fonte, 0.5, cor, 2)
                
                # Mostra o nome detectado
                texto_nome = f"Nome: {nome}"
                cv2.putText(frame, texto_nome, (x, y-25), fonte, 0.5, cor, 2)
                
                # Mostra a pontuação de confiança
                texto_confianca = f"Confianca: {confianca:.1f}"
                cv2.putText(frame, texto_confianca, (x, y-40), fonte, 0.5, cor, 2)
                
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

if __name__ == "__main__":
    try:
        iniciar_reconhecimento()
    except Exception as e:
        print(f"Erro ao iniciar o reconhecimento: {str(e)}")
