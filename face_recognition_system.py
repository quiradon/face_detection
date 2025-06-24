import cv2
import face_recognition
import numpy as np
import json
from datetime import datetime
import os


class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.tolerance = 0.6
        self.faces_data_file = "faces_data.json"
        self.logs_file = "access_logs.json"
        self.load_faces_data()

    def load_faces_data(self):
        """Carrega os dados dos rostos conhecidos do arquivo JSON"""
        if os.path.exists(self.faces_data_file):
            try:
                with open(self.faces_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for person in data:
                        # Converte a codificação de volta para numpy array
                        encoding = np.array(person['encoding'])
                        self.known_faces.append(encoding)
                        self.known_names.append(person['name'])
                print(f"Carregados {len(self.known_names)} rostos conhecidos")
            except Exception as e:
                print(f"Erro ao carregar dados dos rostos: {e}")

    def save_faces_data(self):
        """Salva os dados dos rostos conhecidos no arquivo JSON"""
        try:
            data = []
            for i, name in enumerate(self.known_names):
                data.append({
                    'name': name,
                    'encoding': self.known_faces[i].tolist()
                })
            with open(self.faces_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("Dados dos rostos salvos com sucesso")
        except Exception as e:
            print(f"Erro ao salvar dados dos rostos: {e}")

    def log_access(self, name, status, confidence=None):
        """Registra tentativa de acesso no log"""
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

    def add_new_face(self, image_path, person_name):
        """Adiciona um novo rosto ao sistema"""
        try:
            # Carrega a imagem
            image = face_recognition.load_image_file(image_path)
            
            # Encontra as codificações dos rostos na imagem
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                return False, "Nenhum rosto encontrado na imagem"
            
            if len(face_encodings) > 1:
                return False, "Múltiplos rostos encontrados. Use uma imagem com apenas um rosto"
            
            # Adiciona o novo rosto
            face_encoding = face_encodings[0]
            self.known_faces.append(face_encoding)
            self.known_names.append(person_name)
            
            # Salva os dados
            self.save_faces_data()
            
            return True, f"Rosto de {person_name} adicionado com sucesso"
            
        except Exception as e:
            return False, f"Erro ao processar imagem: {str(e)}"

    def recognize_face(self, frame):
        """Reconhece rostos em um frame da câmera"""
        # Redimensiona o frame para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Encontra os rostos no frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        results = []
        
        for face_encoding in face_encodings:
            # Compara com os rostos conhecidos
            matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=self.tolerance)
            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            
            name = "Desconhecido"
            confidence = 0
            access_granted = False
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    access_granted = True
            
            results.append({
                'name': name,
                'confidence': confidence,
                'access_granted': access_granted
            })
        
        return results, face_locations

    def remove_face(self, person_name):
        """Remove um rosto do sistema"""
        try:
            if person_name in self.known_names:
                index = self.known_names.index(person_name)
                del self.known_names[index]
                del self.known_faces[index]
                self.save_faces_data()
                return True, f"Rosto de {person_name} removido com sucesso"
            else:
                return False, f"Pessoa {person_name} não encontrada"
        except Exception as e:
            return False, f"Erro ao remover rosto: {str(e)}"

    def get_registered_faces(self):
        """Retorna lista de rostos cadastrados"""
        return self.known_names.copy()
