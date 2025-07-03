import cv2
import os
import numpy as np
from PIL import Image
import random
from scipy import ndimage
from tqdm import tqdm  # Para barra de progresso

# Funções de aumento de dados
def aplicar_ruido(imagem, intensidade=0.01):
    """Adiciona ruído gaussiano à imagem"""
    ruido = np.random.normal(0, intensidade, imagem.shape)
    imagem_com_ruido = imagem + ruido * 255
    return np.clip(imagem_com_ruido, 0, 255).astype(np.uint8)

def aplicar_rotacao(imagem, angulo):
    """Rotaciona a imagem por um ângulo específico"""
    return ndimage.rotate(imagem, angulo, reshape=False)

def ajustar_brilho(imagem, fator):
    """Ajusta o brilho da imagem"""
    imagem_ajustada = imagem * fator
    return np.clip(imagem_ajustada, 0, 255).astype(np.uint8)

def aplicar_filtro_bilateral(imagem):
    """Aplica filtro bilateral para reduzir ruído preservando bordas"""
    return cv2.bilateralFilter(imagem, 9, 75, 75)

def aplicar_clahe(imagem):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(imagem)

def aumentar_dados(face_original):
    """Gera variações da imagem original"""
    faces_aumentadas = []
    
    # Aplica pré-processamento na imagem original
    face_preprocessada = aplicar_filtro_bilateral(face_original)
    face_preprocessada = aplicar_clahe(face_preprocessada)
    faces_aumentadas.append(face_preprocessada)
    
    # Gera variações com rotações
    for angulo in [-7, -3, 3, 7]:
        face_rotacionada = aplicar_rotacao(face_preprocessada, angulo)
        faces_aumentadas.append(face_rotacionada)
    
    # Variações de brilho
    for fator in [0.85, 1.15]:
        face_brilho = ajustar_brilho(face_preprocessada, fator)
        faces_aumentadas.append(face_brilho)
    
    return faces_aumentadas

# Função para obter imagens e labels para treinamento
def get_imagens_e_labels(path):
    # Lista para armazenar as faces
    faces = []
    # Lista para armazenar os IDs
    ids = []
    # Lista para armazenar os nomes
    nomes = {}
    id_atual = 0

    # Percorre todas as pastas de usuários
    for usuario in os.listdir(path):
        caminho_usuario = os.path.join(path, usuario)
        if os.path.isdir(caminho_usuario):
            # Associa um ID numérico ao nome do usuário
            nomes[id_atual] = usuario
            
            # Percorre todas as imagens do usuário
            imagens = [img for img in os.listdir(caminho_usuario) 
                      if img.endswith(('.png', '.jpg', '.jpeg'))]
            
            for imagem in tqdm(imagens, desc=f"Processando {usuario}", unit="img"):
                caminho_imagem = os.path.join(caminho_usuario, imagem)
                
                # Carrega a imagem e converte para escala de cinza
                face_img = Image.open(caminho_imagem).convert('L')
                # Redimensiona para 100x100 pixels para reduzir o tamanho do modelo
                face_img = face_img.resize((250, 250), Image.Resampling.LANCZOS)
                face_np = np.array(face_img, 'uint8')
                
                # Aplica equalização de histograma
                face_np = cv2.equalizeHist(face_np)
                
                # Normalização do contraste
                face_np = cv2.normalize(face_np, None, 0, 255, cv2.NORM_MINMAX)
                
                # Gera variações da imagem
                faces_aumentadas = aumentar_dados(face_np)
                
                # Adiciona todas as variações às listas
                for face_aumentada in faces_aumentadas:
                    faces.append(face_aumentada)
                    ids.append(id_atual)
                    
            id_atual += 1
    
    return faces, ids, nomes

# Diretório onde estão as pastas dos usuários com as fotos
path = 'users'

print("\n=== Iniciando Processo de Treinamento ===\n")

# Conta total de imagens para a barra de progresso
total_imagens = sum(len([f for f in os.listdir(os.path.join(path, d)) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
                    for d in os.listdir(path) 
                    if os.path.isdir(os.path.join(path, d)))

print(f"Total de imagens originais encontradas: {total_imagens}")
print(f"Total de imagens após aumentação (7x): {total_imagens * 7}")
print("\n1. Processando e aumentando imagens...")

# Obtém as faces e IDs
faces, ids, nomes = get_imagens_e_labels(path)

print(f"\n2. Iniciando treinamento com {len(faces)} imagens...")
print("Isso pode levar alguns minutos, por favor aguarde...")

# Cria o reconhecedor LBPH com parâmetros otimizados para eficiência
reconhecedor = cv2.face.LBPHFaceRecognizer_create(
    radius=2,           # Raio menor para processamento mais rápido
    neighbors=8,        # Número padrão de pontos de amostra
    grid_x=8,          # Grade padrão para eficiência
    grid_y=8,          # Grade padrão para eficiência
    threshold=110.0     # Mantém o limiar para confiabilidade
)

# Treina o reconhecedor
print("\nTreinando o modelo... (esta etapa pode demorar)")
reconhecedor.train(faces, np.array(ids))
print("Treinamento do modelo concluído!")

print("\n3. Salvando arquivos...")
# Salva o modelo treinado
reconhecedor.write('classificador.yml')
print("✓ Modelo salvo em classificador.yml")

# Salva o dicionário de nomes
import json
with open('nomes.json', 'w') as f:
    json.dump(nomes, f)
print("✓ Mapeamento de nomes salvo em nomes.json")

print("Treinamento concluído com sucesso!")
print(f"Total de pessoas cadastradas: {len(nomes)}")
print("Nomes cadastrados:")
for id_usuario, nome in nomes.items():
    print(f"ID {id_usuario}: {nome}")
