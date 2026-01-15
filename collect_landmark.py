"""
Script pour collecter les landmarks MediaPipe à partir des images
Crée un dataset de coordonnées (x,y,z) pour chaque signe
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle

# Configuration
DATASET_PATH = 'dataset/train'
OUTPUT_PATH = 'model/landmarks_dataset.pkl'
IMG_SIZE = 224

class LandmarkCollector:
    """
    Collecte les landmarks MediaPipe depuis les images
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image_path):
        """
        Extrait les landmarks d'une image
        
        Returns:
            landmarks: Array numpy de shape (63,) contenant x,y,z pour 21 points
                      ou None si pas de main détectée
        """
        # Lire l'image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convertir en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détecter les mains
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extraire les landmarks (21 points × 3 coordonnées = 63 valeurs)
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def collect_from_dataset(self):
        """
        Collecte tous les landmarks du dataset
        """
        print("=" * 60)
        print("   📊 COLLECTE DES LANDMARKS")
        print("=" * 60 + "\n")
        
        # Lister toutes les classes
        classes = sorted([d for d in os.listdir(DATASET_PATH) 
                         if os.path.isdir(os.path.join(DATASET_PATH, d))])
        
        print(f"✅ {len(classes)} classes détectées")
        print(f"📋 Classes: {classes}\n")
        
        # Dictionnaire pour stocker les données
        data = {
            'landmarks': [],
            'labels': [],
            'class_names': classes
        }
        
        total_images = 0
        total_detected = 0
        
        # Pour chaque classe
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(DATASET_PATH, class_name)
            
            # Lister toutes les images
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\n🔄 Traitement de '{class_name}': {len(images)} images")
            
            class_detected = 0
            
            # Traiter chaque image avec barre de progression
            for img_name in tqdm(images, desc=f"  {class_name}", leave=False):
                img_path = os.path.join(class_path, img_name)
                
                # Extraire les landmarks
                landmarks = self.extract_landmarks(img_path)
                
                if landmarks is not None:
                    data['landmarks'].append(landmarks)
                    data['labels'].append(class_idx)
                    class_detected += 1
                
                total_images += 1
            
            total_detected += class_detected
            detection_rate = (class_detected / len(images)) * 100
            print(f"  ✅ {class_detected}/{len(images)} détectés ({detection_rate:.1f}%)")
        
        # Convertir en arrays numpy
        data['landmarks'] = np.array(data['landmarks'])
        data['labels'] = np.array(data['labels'])
        
        print("\n" + "=" * 60)
        print("   📊 RÉSUMÉ")
        print("=" * 60)
        print(f"\n✅ Images totales: {total_images}")
        print(f"✅ Mains détectées: {total_detected} ({(total_detected/total_images)*100:.1f}%)")
        print(f"✅ Shape landmarks: {data['landmarks'].shape}")
        print(f"✅ Shape labels: {data['labels'].shape}")
        
        # Sauvegarder
        os.makedirs('model', exist_ok=True)
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✅ Dataset sauvegardé dans: {OUTPUT_PATH}")
        
        return data
    
    def close(self):
        self.hands.close()

def main():
    """
    Fonction principale
    """
    print("\n" + "🤟 " * 30)
    print("\n   COLLECTE DES LANDMARKS MEDIAPIPE\n")
    print("🤟 " * 30 + "\n")
    
    try:
        collector = LandmarkCollector()
        data = collector.collect_from_dataset()
        collector.close()
        
        print("\n🎉 Collecte terminée avec succès!")
        print("\n💡 Étape suivante:")
        print("   py -3.11 train_classifier.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()