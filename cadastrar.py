# Importa as bibliotecas necessárias
import cv2
import os
import numpy as np

def listar_usuarios(users_folder):
    """
    Lista todos os usuários cadastrados e suas quantidades de fotos.
    """
    if not os.path.exists(users_folder):
        return []
    
    usuarios = []
    for pasta in os.listdir(users_folder):
        pasta_path = os.path.join(users_folder, pasta)
        if os.path.isdir(pasta_path):
            # Conta as fotos do usuário
            fotos = len([f for f in os.listdir(pasta_path) if f.endswith('.png')])
            usuarios.append((pasta, fotos))
    
    return sorted(usuarios)

def mostrar_menu_usuarios(users_folder):
    """
    Mostra o menu de seleção de usuários.
    """
    usuarios = listar_usuarios(users_folder)
    
    print("\n" + "="*60)
    print("🎯 SISTEMA DE CADASTRO PARA RECONHECIMENTO FACIAL")
    print("="*60)
    
    if usuarios:
        print("\n📋 USUÁRIOS CADASTRADOS:")
        for i, (nome, fotos) in enumerate(usuarios, 1):
            print(f"{i:2d}. {nome:<20} ({fotos} foto{'s' if fotos != 1 else ''})")
        
        print(f"\n{len(usuarios) + 1:2d}. [NOVO USUÁRIO]")
        print(f"{len(usuarios) + 2:2d}. [SAIR]")
        
        while True:
            try:
                opcao = input(f"\nEscolha uma opção (1-{len(usuarios) + 2}): ").strip()
                opcao_num = int(opcao)
                
                if 1 <= opcao_num <= len(usuarios):
                    # Usuário existente selecionado
                    nome_selecionado = usuarios[opcao_num - 1][0]
                    fotos_existentes = usuarios[opcao_num - 1][1]
                    print(f"\n✅ Usuário selecionado: {nome_selecionado} ({fotos_existentes} fotos)")
                    return nome_selecionado
                elif opcao_num == len(usuarios) + 1:
                    # Novo usuário
                    novo_nome = input("\n📝 Digite o nome do novo usuário: ").strip()
                    if novo_nome:
                        print(f"\n✅ Novo usuário: {novo_nome}")
                        return novo_nome
                    else:
                        print("❌ Nome não pode estar vazio!")
                elif opcao_num == len(usuarios) + 2:
                    # Sair
                    return None
                else:
                    print("❌ Opção inválida!")
            except ValueError:
                print("❌ Digite apenas números!")
    else:
        print("\n📋 Nenhum usuário cadastrado ainda.")
        print("\n1. [NOVO USUÁRIO]")
        print("2. [SAIR]")
        
        while True:
            try:
                opcao = input("\nEscolha uma opção (1-2): ").strip()
                opcao_num = int(opcao)
                
                if opcao_num == 1:
                    novo_nome = input("\n📝 Digite o nome do novo usuário: ").strip()
                    if novo_nome:
                        print(f"\n✅ Novo usuário: {novo_nome}")
                        return novo_nome
                    else:
                        print("❌ Nome não pode estar vazio!")
                elif opcao_num == 2:
                    return None
                else:
                    print("❌ Opção inválida!")
            except ValueError:
                print("❌ Digite apenas números!")

def main():
    """
    Função principal para capturar e exibir o feed da webcam.
    """
    # Carrega o classificador para detecção facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Cria a pasta 'users' se ela não existir
    users_folder = "users"
    if not os.path.exists(users_folder):
        os.makedirs(users_folder)
    
    # Mostra o menu de seleção de usuários
    nome_usuario = mostrar_menu_usuarios(users_folder)
    
    if nome_usuario is None:
        print("\n👋 Programa encerrado.")
        return
    
    # Inicializa a captura de vídeo da webcam padrão (geralmente o índice 0)
    # Se você tiver mais de uma webcam, pode precisar usar 1, 2, etc.
    cap = cv2.VideoCapture(0)

    # Verifica se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    # Cria a pasta do usuário se ela não existir
    user_folder = os.path.join(users_folder, nome_usuario)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    # Conta fotos existentes
    existing_photos = len([f for f in os.listdir(user_folder) if f.endswith('.png')])
    
    print(f"\n📷 Modo de captura iniciado para: {nome_usuario}")
    print(f"📊 Fotos existentes: {existing_photos}")
    print("\nControles:")
    print("ESPAÇO - Capturar nova foto")
    print("Q - Sair do programa")
    print("\n* Mantenha o foco na janela do vídeo para usar os controles *")

    # Loop infinito para ler os frames da webcam
    while True:
        # Lê um único frame da webcam
        # 'ret' é um booleano (True/False) que indica se a leitura foi bem-sucedida
        # 'frame' é a imagem (array NumPy) capturada
        ret, frame = cap.read()

        # Se a leitura falhar (ex: webcam desconectada), encerra o loop
        if not ret:
            print("Erro: Não foi possível ler o frame da webcam.")
            break

        # Exibe o frame em uma janela chamada "Webcam Feed"
        cv2.imshow('Webcam Feed', frame)

        # Aguarda por 1 milissegundo por uma tecla ser pressionada
        # O `& 0xFF` é uma máscara para garantir compatibilidade entre sistemas
        key = cv2.waitKey(1) & 0xFF
        
        # Se a tecla pressionada for 'q', o loop é interrompido
        if key == ord('q'):
            break
        # Se a tecla pressionada for ESPAÇO (código 32), tira uma foto
        elif key == 32:  # Código ASCII do ESPAÇO
            # Captura a foto
            captured_photo = frame.copy()
            
            # Converte para escala de cinza para detecção facial
            gray = cv2.cvtColor(captured_photo, cv2.COLOR_BGR2GRAY)
            
            # Detecta faces na imagem
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)  # Tamanho mínimo da face
            )
            
            if len(faces) == 0:
                # Se nenhuma face for detectada, mostra mensagem e continua
                error_image = captured_photo.copy()
                cv2.putText(error_image, "NENHUM ROSTO DETECTADO!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_image, "Tente novamente...", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Webcam Feed', error_image)
                cv2.waitKey(2000)  # Mostra por 2 segundos
                continue
            elif len(faces) > 1:
                # Se mais de uma face for detectada, mostra mensagem e continua
                error_image = captured_photo.copy()
                cv2.putText(error_image, "MULTIPLOS ROSTOS DETECTADOS!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_image, "Fotografe apenas uma pessoa por vez...", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Webcam Feed', error_image)
                cv2.waitKey(2000)  # Mostra por 2 segundos
                continue
            
            # Pega a primeira (e única) face detectada
            (x, y, w, h) = faces[0]
            
            # Adiciona uma margem ao redor do rosto (20% de cada lado)
            margin = int(0.2 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(captured_photo.shape[1] - x, w + 2*margin)
            h = min(captured_photo.shape[0] - y, h + 2*margin)
            
            # Recorta apenas a região do rosto com a margem
            face_img = captured_photo[y:y+h, x:x+w]

            # Redimensiona para um tamanho padrão (por exemplo, 300x300)
            face_img = cv2.resize(face_img, (300, 300))
            
            # Atualiza a foto capturada para conter apenas o rosto
            captured_photo = face_img.copy()
            
            # Cria uma cópia da foto para adicionar texto
            photo_with_text = captured_photo.copy()
            
            # Adiciona texto na imagem
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 255, 0)  # Verde
            thickness = 2
            
            # Texto principal
            text1 = "FOTO CAPTURADA!"
            text2 = f"Usuario: {nome_usuario}"
            text3 = "1 - SIM (salvar)    2 - NAO (descartar)"
            
            # Calcula posições do texto
            (w, h) = photo_with_text.shape[:2][::-1]
            
            # Posições do texto
            y_start = 50
            cv2.putText(photo_with_text, text1, (50, y_start), font, font_scale, color, thickness)
            cv2.putText(photo_with_text, text2, (50, y_start + 40), font, 0.6, (255, 255, 255), 2)
            cv2.putText(photo_with_text, text3, (50, y_start + 80), font, 0.6, (255, 255, 0), 2)
            
            # Exibe a foto com o texto
            cv2.imshow('Webcam Feed', photo_with_text)
            
            # Loop para aguardar a resposta do usuário
            while True:
                key_choice = cv2.waitKey(0) & 0xFF
                
                if key_choice == ord('1'):
                    # Pergunta o nome da pessoa na tela
                    name_input_image = captured_photo.copy()
                    cv2.putText(name_input_image, "(Digite no terminal e pressione ENTER)", (50, 90), font, 0.5, (255, 255, 255), 2)
                    cv2.imshow('Webcam Feed', name_input_image)
                    cv2.waitKey(1)  # Atualiza a tela
                    
                    # Captura o nome no terminal
                    print("\n" + "="*50)
                    print("📝 CADASTRO DE USUÁRIO")
                    print("="*50)
                    nome_pessoa = nome_usuario
                    
                    if nome_pessoa:
                        # Cria uma pasta para o usuário se ela não existir
                        user_folder = os.path.join(users_folder, nome_pessoa)
                        if not os.path.exists(user_folder):
                            os.makedirs(user_folder)
                        
                        # Conta quantas fotos já existem para este usuário
                        existing_photos = len([f for f in os.listdir(user_folder) if f.endswith('.png')])
                        photo_number = existing_photos + 1
                        
                        # Salva a foto na pasta do usuário
                        filename = f"{nome_pessoa}_{photo_number:03d}.png"
                        filepath = os.path.join(user_folder, filename)
                        cv2.imwrite(filepath, captured_photo)
                        
                        # Mostra confirmação na tela
                        confirm_image = captured_photo.copy()
                        cv2.putText(confirm_image, f"FOTO {photo_number} SALVA!", (50, 50), font, 0.8, (0, 255, 0), 2)
                        cv2.putText(confirm_image, f"Usuario: {nome_pessoa}", (50, 90), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(confirm_image, f"Arquivo: {filename}", (50, 130), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(confirm_image, "ESPACO - Tirar mais fotos", (50, 170), font, 0.5, (0, 255, 255), 2)
                        cv2.putText(confirm_image, "Pressione qualquer tecla para continuar", (50, 210), font, 0.5, (200, 200, 200), 2)
                        cv2.imshow('Webcam Feed', confirm_image)
                        print(f"✅ Foto {photo_number} do usuário '{nome_pessoa}' salva!")
                        print(f"📁 Caminho: {filepath}")
                        print(f"� Total de fotos do usuário: {photo_number}")
                        cv2.waitKey(0)
                    else:
                        # Se não digitou nome, cancela
                        cancel_image = captured_photo.copy()
                        cv2.putText(cancel_image, "CADASTRO CANCELADO", (50, 50), font, 0.8, (0, 0, 255), 2)
                        cv2.putText(cancel_image, "Nome nao informado", (50, 90), font, 0.6, (255, 255, 255), 2)
                        cv2.imshow('Webcam Feed', cancel_image)
                        print("❌ Cadastro cancelado - nome não informado")
                        cv2.waitKey(2000)  # Mostra por 2 segundos
                    break
                elif key_choice == ord('2'):
                    # Mostra confirmação de descarte na tela
                    discard_image = captured_photo.copy()
                    cv2.putText(discard_image, "FOTO DESCARTADA", (50, 50), font, 1, (0, 0, 255), 3)
                    cv2.putText(discard_image, "Voltando para a camera...", (50, 100), font, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Webcam Feed', discard_image)
                    cv2.waitKey(1000)  # Mostra por 1 segundo
                    break
                elif key_choice == ord('q'):
                    # Se pressionar 'q', sai do programa
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Feed encerrado.")
                    return

    # Após sair do loop, libera os recursos da webcam
    cap.release()
    # Fecha todas as janelas abertas pelo OpenCV
    cv2.destroyAllWindows()
    print("Feed encerrado.")

if __name__ == '__main__':
    main()