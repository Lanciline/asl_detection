"""
Détection en temps réel basée sur les landmarks MediaPipe
Beaucoup plus rapide et précis que l'approche CNN
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
import time

# Configuration
MODEL_PATH = 'model/landmark_classifier.pkl'
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_FRAMES = 7

class LandmarkASLDetector:
    """
    Détecteur ASL basé sur les landmarks MediaPipe
    """
    
    def __init__(self):
        print("🚀 Initialisation du détecteur ASL (Landmarks)...")
        
        # Charger le modèle
        try:
            model_data = joblib.load(MODEL_PATH)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.class_names = model_data['class_names']
            self.use_scaler = model_data['use_scaler']
            print(f"✅ Modèle {model_data['model_type']} chargé")
            print(f"✅ {len(self.class_names)} classes: {self.class_names}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise
        
        # Initialiser MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # File pour lissage
        self.prediction_queue = deque(maxlen=SMOOTHING_FRAMES)
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Détecteur initialisé\n")
    
    def extract_landmarks(self, hand_landmarks):
        """
        Extrait les landmarks en array numpy
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).reshape(1, -1)
    
    def predict(self, landmarks):
        """
        Prédit la classe à partir des landmarks
        """
        # Normaliser si nécessaire
        if self.use_scaler and self.scaler is not None:
            landmarks = self.scaler.transform(landmarks)
        
        # Prédire
        prediction = self.classifier.predict(landmarks)[0]
        
        # Obtenir les probabilités si disponible
        if hasattr(self.classifier, 'predict_proba'):
            probas = self.classifier.predict_proba(landmarks)[0]
            confidence = probas[prediction]
        else:
            # Pour Random Forest sans proba
            confidence = 0.8  # Valeur par défaut
        
        return prediction, confidence
    
    def get_smoothed_prediction(self, prediction):
        """
        Lisse les prédictions
        """
        self.prediction_queue.append(prediction)
        
        if len(self.prediction_queue) > 0:
            most_common = max(set(self.prediction_queue), 
                            key=list(self.prediction_queue).count)
            return most_common
        
        return prediction
    
    def draw_info(self, frame, class_name, confidence, bbox=None):
        """
        Affiche les informations sur la frame
        """
        h, w, _ = frame.shape
        
        # Dessiner le bounding box si disponible
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Préparer le texte
        if confidence > CONFIDENCE_THRESHOLD and class_name != 'nothing':
            text = f"{class_name.upper()}"
            conf_text = f"{confidence:.0%}"
            color = (0, 255, 0)
            status = "DETECTE"
        else:
            text = "..."
            conf_text = ""
            color = (100, 100, 100)
            status = "EN ATTENTE"
        
        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texte principal (lettre)
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   2.0, color, 4)
        
        # Confiance
        if conf_text:
            cv2.putText(frame, conf_text, (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
        
        # Status
        cv2.putText(frame, status, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (200, 200, 200), 1)
        
        # FPS
        fps_bg = frame.copy()
        cv2.rectangle(fps_bg, (w-180, 10), (w-10, 60), (0, 0, 0), -1)
        cv2.addWeighted(fps_bg, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-170, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Q = Quitter  |  SPACE = Pause", (15, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Barre de confiance
        if confidence > 0:
            bar_width = int(300 * confidence)
            bar_color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (20, 135), (20 + bar_width, 150), bar_color, -1)
            cv2.rectangle(frame, (20, 135), (320, 150), (255, 255, 255), 2)
    
    def get_bounding_box(self, hand_landmarks, frame_shape):
        """
        Calcule le bounding box de la main
        """
        h, w, _ = frame_shape
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        margin = 0.1
        x_min = max(0, int((min(x_coords) - margin) * w))
        x_max = min(w, int((max(x_coords) + margin) * w))
        y_min = max(0, int((min(y_coords) - margin) * h))
        y_max = min(h, int((max(y_coords) + margin) * h))
        
        return (x_min, y_min, x_max, y_max)
    
    def update_fps(self):
        """
        Met à jour le FPS
        """
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            elapsed = time.time() - self.start_time
            self.fps = 20 / elapsed
            self.start_time = time.time()
    
    def run(self):
        """
        Boucle principale
        """
        print("📹 Démarrage de la caméra...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Caméra démarrée")
        print("\n🎯 Détection en cours...")
        print("   Montrez votre signe à la caméra")
        print("   Q = Quitter | SPACE = Pause\n")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Effet miroir
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Détecter les mains
                results = self.hands.process(rgb_frame)
                
                class_name = "nothing"
                confidence = 0.0
                bbox = None
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Dessiner les landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Extraire landmarks
                        landmarks = self.extract_landmarks(hand_landmarks)
                        
                        # Prédire
                        prediction, confidence = self.predict(landmarks)
                        
                        # Lisser
                        smoothed_pred = self.get_smoothed_prediction(prediction)
                        class_name = self.class_names[smoothed_pred]
                        
                        # Bounding box
                        bbox = self.get_bounding_box(hand_landmarks, frame.shape)
                
                # Afficher infos
                self.draw_info(frame, class_name, confidence, bbox)
                self.update_fps()
                
                cv2.imshow('ASL Landmark Detection', frame)
            else:
                # Mode pause
                pause_frame = frame.copy()
                overlay = pause_frame.copy()
                h, w = pause_frame.shape[:2]
                cv2.rectangle(overlay, (w//2-150, h//2-50), (w//2+150, h//2+50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.8, pause_frame, 0.2, 0, pause_frame)
                cv2.putText(pause_frame, "PAUSE", (w//2-80, h//2+10), 
                           cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3)
                cv2.imshow('ASL Landmark Detection', pause_frame)
            
            # Touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸️  PAUSE" if paused else "▶️  REPRISE")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✅ Détection terminée")

def main():
    """
    Fonction principale
    """
    print("\n" + "=" * 60)
    print("   🤟 ASL Detection (Landmarks MediaPipe)")
    print("=" * 60 + "\n")
    
    try:
        detector = LandmarkASLDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interruption")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()