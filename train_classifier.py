"""
Script d'entraînement d'un classificateur basé sur les landmarks MediaPipe
Beaucoup plus rapide et précis que le CNN sur images
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration
LANDMARKS_PATH = 'model/landmarks_dataset.pkl'
MODEL_PATH = 'model/landmark_classifier.pkl'
SCALER_PATH = 'model/landmark_scaler.pkl'

def load_data():
    """
    Charge les données de landmarks
    """
    print("📂 Chargement des données...")
    with open(LANDMARKS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    X = data['landmarks']
    y = data['labels']
    class_names = data['class_names']
    
    print(f"✅ {len(X)} échantillons chargés")
    print(f"✅ {len(class_names)} classes")
    print(f"✅ Shape: {X.shape}")
    
    return X, y, class_names

def train_random_forest(X_train, y_train, X_test, y_test, class_names):
    """
    Entraîne un Random Forest (rapide et efficace)
    """
    print("\n" + "=" * 60)
    print("   🌲 RANDOM FOREST CLASSIFIER")
    print("=" * 60 + "\n")
    
    print("🔄 Entraînement en cours...")
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    clf.fit(X_train, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Précision: {accuracy*100:.2f}%")
    
    return clf, accuracy

def train_mlp(X_train, y_train, X_test, y_test, class_names):
    """
    Entraîne un réseau de neurones MLP (plus précis)
    """
    print("\n" + "=" * 60)
    print("   🧠 RÉSEAU DE NEURONES (MLP)")
    print("=" * 60 + "\n")
    
    # Normalisation des données
    print("🔄 Normalisation...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🔄 Entraînement en cours...")
    
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Précision: {accuracy*100:.2f}%")
    
    return clf, scaler, accuracy

def plot_confusion_matrix(y_test, y_pred, class_names):
    """
    Affiche la matrice de confusion
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=150)
    print("📊 Matrice de confusion sauvegardée: model/confusion_matrix.png")

def print_classification_report(y_test, y_pred, class_names):
    """
    Affiche le rapport de classification détaillé
    """
    print("\n" + "=" * 60)
    print("   📊 RAPPORT DE CLASSIFICATION")
    print("=" * 60 + "\n")
    
    report = classification_report(y_test, y_pred, 
                                   target_names=class_names,
                                   zero_division=0)
    print(report)

def main():
    """
    Fonction principale
    """
    print("\n" + "🎯 " * 30)
    print("\n   ENTRAÎNEMENT DU CLASSIFICATEUR LANDMARKS\n")
    print("🎯 " * 30 + "\n")
    
    try:
        # Charger les données
        X, y, class_names = load_data()
        
        # Split train/test
        print("\n🔀 Séparation train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train: {len(X_train)} échantillons")
        print(f"✅ Test: {len(X_test)} échantillons")
        
        # Entraîner Random Forest
        rf_clf, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, class_names)
        
        # Entraîner MLP
        mlp_clf, scaler, mlp_acc = train_mlp(X_train, y_train, X_test, y_test, class_names)
        
        # Choisir le meilleur
        print("\n" + "=" * 60)
        print("   🏆 COMPARAISON DES MODÈLES")
        print("=" * 60 + "\n")
        
        print(f"Random Forest: {rf_acc*100:.2f}%")
        print(f"MLP (Neural Net): {mlp_acc*100:.2f}%")
        
        if mlp_acc >= rf_acc:
            print("\n✅ MLP sélectionné (meilleure précision)")
            best_clf = mlp_clf
            best_name = "MLP"
            use_scaler = True
            
            # Normaliser aussi les données de test pour les graphiques
            X_test_final = scaler.transform(X_test)
        else:
            print("\n✅ Random Forest sélectionné (meilleure précision)")
            best_clf = rf_clf
            best_name = "RandomForest"
            use_scaler = False
            X_test_final = X_test
            scaler = None
        
        # Prédictions finales
        y_pred = best_clf.predict(X_test_final)
        
        # Rapport de classification
        print_classification_report(y_test, y_pred, class_names)
        
        # Matrice de confusion
        plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Sauvegarder le modèle
        model_data = {
            'classifier': best_clf,
            'scaler': scaler,
            'class_names': class_names,
            'model_type': best_name,
            'use_scaler': use_scaler
        }
        
        joblib.dump(model_data, MODEL_PATH)
        print(f"\n✅ Modèle sauvegardé: {MODEL_PATH}")
        
        # Résumé final
        print("\n" + "=" * 60)
        print("   🎉 ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)
        print(f"\n📊 Modèle: {best_name}")
        print(f"📊 Précision: {max(rf_acc, mlp_acc)*100:.2f}%")
        print(f"📊 Classes: {len(class_names)}")
        
        print("\n💡 Étape suivante:")
        print("   py -3.11 live_detection_landmarks.py")
        
    except FileNotFoundError:
        print("❌ Fichier landmarks_dataset.pkl non trouvé!")
        print("💡 Lancez d'abord: py -3.11 collect_landmarks.py")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()