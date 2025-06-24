import cv2
import numpy as np
import json
from datetime import datetime
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class SimpleFaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.tolerance = 0.75  # Para similaridade de coseno
        self.faces_data_file = "faces_data.pkl"
        self.logs_file = "access_logs.json"
        
        # Inicializar detector de face do OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.load_faces_data()

    def extract_face_features(self, face_image):
        """Extrai características da face usando histogramas e textura"""
        # Converter para escala de cinza se necessário
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Redimensionar para tamanho padrão
        gray = cv2.resize(gray, (100, 100))
        
        # Normalizar iluminação
        gray = cv2.equalizeHist(gray)
        
        # Extrair histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extrair características LBP (Local Binary Pattern) simplificado
        lbp_features = self.extract_lbp_features(gray)
        
        # Extrair características de gradiente
        gradient_features = self.extract_gradient_features(gray)
        
        # Combinar características
        features = np.concatenate([hist, lbp_features, gradient_features])
        
        return features

    def extract_lbp_features(self, gray_image):
        """Extrai características Local Binary Pattern simplificado"""
        h, w = gray_image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                binary = 0
                powers = [1, 2, 4, 8, 16, 32, 64, 128]
                
                # Comparar com os 8 vizinhos
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary += powers[k]
                
                lbp[i-1, j-1] = binary
        
        # Calcular histograma do LBP
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
        
        return hist_lbp

    def extract_gradient_features(self, gray_image):
        """Extrai características de gradiente"""
        # Calcular gradientes
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Histograma da magnitude
        hist_mag = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [64], [0, 256])
        hist_mag = cv2.normalize(hist_mag, hist_mag).flatten()
        
        return hist_mag

    def load_faces_data(self):
        """Carrega os dados dos rostos conhecidos do arquivo pickle"""
        if os.path.exists(self.faces_data_file):
            try:
                with open(self.faces_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                print(f"Carregados {len(self.known_names)} rostos conhecidos")
            except Exception as e:
                print(f"Erro ao carregar dados dos rostos: {e}")
                self.known_faces = []
                self.known_names = []

    def save_faces_data(self):
        """Salva os dados dos rostos conhecidos no arquivo pickle"""
        try:
            data = {
                'faces': self.known_faces,
                'names': self.known_names
            }
            with open(self.faces_data_file, 'wb') as f:
                pickle.dump(data, f)
            print("Dados dos rostos salvos com sucesso")
        except Exception as e:
            print(f"Erro ao salvar dados dos rostos: {e}")

    def log_access(self, name, status, confidence=None):
        """Registra tentativa de acesso no log"""
        # Converter confidence para float puro, mesmo se vier como array numpy ou tipo estranho
        if confidence is not None:
            try:
                # Se for array numpy, pegar o primeiro elemento
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                confidence = float(confidence)
            except Exception:
                confidence = None
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'status': status,  # 'PERMITIDO' ou 'NEGADO'
            'confidence': confidence
        }
        
        logs = []
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        # Manter apenas os últimos 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        try:
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar log: {e}")

    def get_logs(self):
        """Retorna todos os logs de acesso"""
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def detect_faces(self, image):
        """Detecta faces na imagem usando Haar Cascades com parâmetros otimizados e filtro de proporção"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Parâmetros mais rigorosos para precisão
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,  # menor que 1.1 para mais precisão
                minNeighbors=7,    # mais vizinhos para menos falsos positivos
                minSize=(60, 60),  # rostos menores são ignorados
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            face_regions = []
            for (x, y, w, h) in faces:
                # Filtro de proporção (descarta regiões muito largas/estreitas)
                aspect = w / float(h)
                if aspect < 0.7 or aspect > 1.4:
                    continue
                # Filtro de tamanho mínimo absoluto
                if w < 60 or h < 60:
                    continue
                face_region = image[y:y+h, x:x+w]
                face_regions.append({
                    'face': face_region,
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 1.0  # Haar cascades não fornecem confiança
                })
            return face_regions
        except Exception as e:
            print(f"Erro na detecção de faces: {e}")
            return []

    def add_new_face(self, image_path, person_name):
        """Adiciona um novo rosto ao sistema"""
        try:
            # Carrega a imagem
            image = cv2.imread(image_path)
            
            if image is None:
                return False, "Não foi possível carregar a imagem"
            
            # Detecta faces na imagem
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                return False, "Nenhuma face encontrada na imagem"
            
            if len(faces) > 1:
                return False, "Múltiplas faces encontradas. Use uma imagem com apenas uma face"
            
            # Extrair características da face
            face_features = self.extract_face_features(faces[0]['face'])
            
            # Verificar se já existe e remover se necessário
            if person_name in self.known_names:
                self.remove_face(person_name)
            
            # Adicionar nova face
            self.known_faces.append(face_features)
            self.known_names.append(person_name)
            
            # Salvar os dados
            self.save_faces_data()
            
            return True, f"Face de {person_name} adicionada com sucesso"
            
        except Exception as e:
            return False, f"Erro ao processar imagem: {str(e)}"

    def recognize_faces(self, frame):
        """Reconhece faces em um frame da câmera"""
        try:
            # Detectar faces usando Haar Cascades
            detected_faces = self.detect_faces(frame)
            
            results = []
            
            for face_data in detected_faces:
                face_region = face_data['face']
                bbox = face_data['bbox']
                
                # Extrair características da face detectada
                face_features = self.extract_face_features(face_region)
                
                name = "Desconhecido"
                confidence = 0
                access_granted = False
                
                if len(self.known_faces) > 0:
                    # Comparar com faces conhecidas usando similaridade de coseno
                    similarities = []
                    for known_face in self.known_faces:
                        similarity = cosine_similarity([face_features], [known_face])[0][0]
                        similarities.append(similarity)
                    
                    max_similarity = max(similarities)
                    
                    if max_similarity >= self.tolerance:
                        best_match_index = similarities.index(max_similarity)
                        name = self.known_names[best_match_index]
                        confidence = max_similarity
                        access_granted = True
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'access_granted': access_granted,
                    'bbox': bbox
                })
            
            return results
            
        except Exception as e:
            print(f"Erro no reconhecimento: {e}")
            return []

    def remove_face(self, person_name):
        """Remove um rosto do sistema"""
        try:
            if person_name in self.known_names:
                index = self.known_names.index(person_name)
                del self.known_names[index]
                del self.known_faces[index]
                self.save_faces_data()
                return True, f"Face de {person_name} removida com sucesso"
            else:
                return False, f"Pessoa {person_name} não encontrada"
        except Exception as e:
            return False, f"Erro ao remover face: {str(e)}"

    def get_registered_faces(self):
        """Retorna lista de faces cadastradas"""
        return self.known_names.copy()


if __name__ == "__main__":
    # Teste básico do sistema
    system = SimpleFaceRecognitionSystem()
    print("Sistema simples de reconhecimento facial inicializado!")
