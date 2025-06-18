import time
import asyncio
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ========== Config ==========
class Config:
    CONF_THRESH = 0.4
    CLASSIF_THRESH = 0.6
    MIN_CROP_SIZE = 50
    DEVICE = "cpu"
    CROPS_DIR = Path("/tmp/pestai_crops")
    MODELE_RAVAGEUR_DIR = Path("models/modele_ravageur")
    DETECTION_PARTS_DIR = MODELE_RAVAGEUR_DIR / "detection"
    CLASSIF_PARTS_DIR = MODELE_RAVAGEUR_DIR / "classification"
    TEMP_MODELS_DIR = Path("/tmp/rebuilt_models")
    SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

CLASSES_RAVAGEUR = [
    "ants", "bees", "beetle", "catterpillar",
    "earthworms", "earwig", "grasshopper",
    "moth", "slug", "snail", "wasp", "weevil"
]

# ========== Pydantic Schemas ==========
class DetectionBox(BaseModel):
    bbox: List[int]
    confidence: float
    label: str
    crop_url: str
    classification: Optional[dict] = None

class DetectionResponse(BaseModel):
    filename: str
    width: int
    height: int
    detections: List[DetectionBox]

# ========== Fonction reconstruction ==========
def rebuild_file_from_parts(parts_dir: Path, filename_prefix: str, final_filepath: Path):
    part_files = sorted(parts_dir.glob(f"{filename_prefix}.part_*"))
    if not part_files:
        raise FileNotFoundError(f"Aucun fichier trouvé avec le préfixe {filename_prefix} dans {parts_dir}")
    
    with open(final_filepath, "wb") as wfd:
        for part in part_files:
            with open(part, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

# ========== FastAPI app ==========
app = FastAPI(title="API Ravageurs", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

models = {}

@app.on_event("startup")
async def startup():
    Config.CROPS_DIR.mkdir(exist_ok=True, parents=True)
    Config.TEMP_MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    detect_model_path = Config.TEMP_MODELS_DIR / "best_detection_ravageur.onnx"
    classif_model_path = Config.TEMP_MODELS_DIR / "best_model.keras"
    
    try:
        rebuild_file_from_parts(Config.DETECTION_PARTS_DIR, "best_detection_ravageur.onnx", detect_model_path)
        rebuild_file_from_parts(Config.CLASSIF_PARTS_DIR, "best_model.keras", classif_model_path)
    except Exception as e:
        print(f"Erreur reconstruction modèles ravageur: {e}")
        raise
    
    try:
        models["detect"] = YOLO(str(detect_model_path), task="detect")
    except Exception as e:
        print(f"Erreur chargement modèle detection ravageur: {e}")
        models["detect"] = None
        
    try:
        models["classif"] = load_model(str(classif_model_path))
    except Exception as e:
        print(f"Erreur chargement modèle classification ravageur: {e}")
        models["classif"] = None
    
    asyncio.create_task(clean_old_files())

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    if file.content_type not in Config.SUPPORTED_IMAGE_TYPES:
        raise HTTPException(415, "Format non supporté")
    
    contents = await file.read()
    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image invalide")

    h, w = img.shape[:2]
    detections = []
    
    if models.get("detect") is None:
        raise HTTPException(500, "Modèle de détection non disponible")
    if models.get("classif") is None:
        raise HTTPException(500, "Modèle de classification non disponible")

    results = models["detect"].predict(img, conf=Config.CONF_THRESH, device=Config.DEVICE)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) < Config.MIN_CROP_SIZE or (y2 - y1) < Config.MIN_CROP_SIZE:
                continue
            
            crop = img[y1:y2, x1:x2]
            crop_path = Config.CROPS_DIR / f"crop_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_url = f"{request.base_url}crop/{crop_path.name}"
            
            input_shape = models["classif"].input_shape[1:3]
            crop_resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), input_shape[::-1])
            crop_input = effnet_preprocess(crop_resized[np.newaxis, ...])
            
            preds = models["classif"].predict(crop_input, verbose=0)
            idx = np.argmax(preds[0])
            conf = float(preds[0][idx])

            detections.append(DetectionBox(
                bbox=[x1, y1, x2, y2],
                confidence=round(float(box.conf[0]), 4),
                label="ravageur",
                crop_url=crop_url,
                classification={"predicted_class": CLASSES_RAVAGEUR[idx], "confidence": round(conf, 4)}
            ))
    
    return DetectionResponse(filename=file.filename, width=w, height=h, detections=detections)

@app.get("/crop/{filename}")
async def get_crop(filename: str):
    path = Config.CROPS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Crop introuvable")
    return FileResponse(path)

async def clean_old_files():
    while True:
        now = time.time()
        for file in Config.CROPS_DIR.glob("*"):
            if file.is_file() and (now - file.stat().st_mtime > 3600):
                try:
                    file.unlink()
                except Exception:
                    pass
        await asyncio.sleep(3600)
