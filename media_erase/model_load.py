import torch
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from loguru import logger
import hashlib
from . util import norm_img, download_model

def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def handle_error(model_path, model_md5, e):
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
            )
        except:
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete {model_path} and restart lama-cleaner."
            )
    else:
        logger.error(
            f"Failed to load model {model_path},"
            f"please submit an issue at https://github.com/Sanster/lama-cleaner/issues and include a screenshot of the error:\n{e}"
        )
    exit(-1)


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model



# 모델 초기화
def load_lama_cleaner():
    print("lama cleaner setup!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LAMA_MODEL_URL = os.environ.get(
        "LAMA_MODEL_URL",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    )
    LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

    lama_model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    
    # 여기서 lame_model pth 로 저장
    return lama_model


def load_yolo():
    # ultralytics 에서 기본으로 제공되는 yolov8s.pt 모델 사용
    yolo_model = YOLO('yolov8s.pt')
    
    return yolo_model
    

