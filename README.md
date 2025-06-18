API unifiée de détection et classification de maladies des plantes et ravageurs, basée sur des modèles YOLO (détection) et Keras (classification). L’API est packagée dans un conteneur Docker pour un déploiement facile.
Structure du projet

.
├── api_pestai_unifiee_docker.py      # Code principal de l’API FastAPI
├── dockerfile                        # Dockerfile pour construire l’image
├── models                           # Modèles ML
│   ├── modele_feuille
│   │   ├── classification
│   │   │   └── model_final.keras
│   │   └── detection
│   │       └── last.onnx
│   └── modele_ravageur
│       ├── classification
│       │   └── best_model.keras
│       └── detection
│           └── best_detection_ravageur.onnx
├── requirements.txt        # Dépendances Python
├── azureml-deploy.yml     #Ce fichier décrit comment Azure ML doit construire et déployer l'image Docker                
└── __pycache__

Prérequis

    Docker installé sur la machine

    Accès à un GPU (optionnel, sinon l’API fonctionne en CPU)

Installation & Lancement
Construction de l’image Docker

docker build -t pestai_api .

Lancement du conteneur

docker run -d -p 8000:8000 --name pestai_api pestai_api

Utilisation

    Endpoint principal :
    POST /detect-image
    Paramètre type : "feuille" ou "ravageur"
    Envoi d’une image (jpeg, png, bmp, tiff, webp) pour détection et classification

    Endpoint récupération crop :
    GET /crop/{filename}
    Récupérer les images recadrées issues des détections
