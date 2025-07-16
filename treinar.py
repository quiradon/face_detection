import cv2
import os
import numpy as np
from PIL import Image
import random
from scipy import ndimage
from tqdm import tqdm  # Para barra de progresso

# Funções de aumento de dados
def aplicar_ruido(imagem, intensidade=0.01):
    # Vamos adicionar ruído gaussiano à imagem para criar variações
    ruido = np.random.normal(0, intensidade, imagem.shape)
    imagem_com_ruido = imagem + ruido * 255
    return np.clip(imagem_com_ruido, 0, 255).astype(np.uint8)

def aplicar_rotacao(imagem, angulo):
    # Vamos rotacionar a imagem por um ângulo específico
    return ndimage.rotate(imagem, angulo, reshape=False)

def ajustar_brilho(imagem, fator):
    # Vamos ajustar o brilho da imagem
    imagem_ajustada = imagem * fator
    return np.clip(imagem_ajustada, 0, 255).astype(np.uint8)

def ajustar_gamma(imagem, gamma=1.0):
    # Vamos ajustar o gamma da imagem para simular diferentes condições de iluminação
    inv_gamma = 1.0 / gamma
    tabela = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(imagem, tabela)

def simular_exposicao(imagem, exposicao=1.0):
    # Simula diferentes exposições de câmera
    exposicao = max(0.1, exposicao)
    # Aplica transformação logarítmica para simular exposição
    imagem_exp = np.log1p(imagem * exposicao / 255.0) * (255.0 / np.log1p(exposicao))
    return np.clip(imagem_exp, 0, 255).astype(np.uint8)

def preprocessar_face(face_img):
    """
    Aplica o mesmo pré-processamento usado na detecção
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
    
    # Aplica pré-processamento consistente na imagem original
    face_preprocessada = preprocessar_face(face_original)
    faces_aumentadas.append(face_preprocessada)
    
    # Gera variações com rotações (limitado para não sobrecarregar)
    for angulo in [-5, 5]:
        face_rotacionada = aplicar_rotacao(face_preprocessada, angulo)
        faces_aumentadas.append(face_rotacionada)
    
    # Variações de gamma (simula diferentes tipos de dispositivos/monitores)
    for gamma in [0.8, 1.2]:
        face_gamma = ajustar_gamma(face_preprocessada, gamma)
        faces_aumentadas.append(face_gamma)
    
    # Variações de exposição (simula diferentes condições de câmera)
    for exp in [0.85, 1.25]:
        face_exp = simular_exposicao(face_preprocessada, exp)
        faces_aumentadas.append(face_exp)
    
    # Variação com combinação estratégica para cenários realistas
    # Combinação de gamma baixo com exposição alta (cenário comum em ambientes reais)
    face_combinada = simular_exposicao(ajustar_gamma(face_preprocessada, 0.85), 1.2)
    faces_aumentadas.append(face_combinada)
    
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
                # Redimensiona para 250x250 pixels (mesmo tamanho usado na detecção)
                face_img = face_img.resize((250, 250), Image.Resampling.LANCZOS)
                face_np = np.array(face_img, 'uint8')
                # Gera variações da imagem (que incluem o pré-processamento)
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
print(f"Total de imagens após aumentação (6x): {total_imagens * 6}")
print("\n1. Processando imagens com pré-processamento consistente (bilateral + CLAHE + equalização)...")

# Obtém as faces e IDs
faces, ids, nomes = get_imagens_e_labels(path)

print(f"\n2. Iniciando treinamento com {len(faces)} imagens...")
print("Isso pode levar alguns minutos, por favor aguarde...")

# Cria o reconhecedor LBPH com parâmetros otimizados
reconhecedor = cv2.face.LBPHFaceRecognizer_create(
    radius=2,           # Raio menor para processamento mais rápido
    neighbors=8,        # Número padrão de pontos de amostra
    grid_x=8,           # Grade padrão para eficiência
    grid_y=8,           # Grade padrão para eficiência
    threshold=100.0     # Limiar ajustado para o pré-processamento consistente
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
# Registra as otimizações aplicadas
import datetime
with open(f'logs/otimizacoes_{datetime.datetime.now().strftime("%Y-%m-%d")}.txt', 'a') as f:
    f.write(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Treinamento concluído: {len(faces)} imagens processadas com pré-processamento consistente.\n")

print("\nTreinamento concluído com sucesso!")
print(f"Total de pessoas cadastradas: {len(nomes)}")
print(f"Total de imagens processadas após aumentação: {len(faces)}")
print("Otimizações aplicadas: pré-processamento consistente entre treinamento e detecção")
print("Filtros aplicados: bilateral + CLAHE + equalização + normalização")
print("\nNomes cadastrados:")
for id_usuario, nome in nomes.items():
    print(f"ID {id_usuario}: {nome}")
