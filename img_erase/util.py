import os
import sys
import cv2

from urllib.parse import urlparse

import numpy as np
import torch
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file

def download_model(url, model_md5: str = None):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                        f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart lama-cleaner."
                    )
                exit(-1)

    return cached_file


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

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

def get_mask(boxes, image_array):
    # mask_image 변환
    # 빈(mask) 이미지 생성

    mask = np.zeros((image_array.shape[0],image_array.shape[1]))

    # bounding box 정보를 사용하여 마스크 이미지를 만듭니다.
    # media/mask_image/ 저장해야 하나??
    
    for (center_x, center_y, width, height) in boxes:
        # bounding box의 좌표 계산
        x = int((center_x - width / 2) * image_array.shape[1])
        y = int((center_y - height / 2) * image_array.shape[0])
        w = int(width * image_array.shape[1])
        h = int(height * image_array.shape[0])
        # 성능향상을 위한 boundig box 크기 조정
        x -= 10
        y -= 20

        # w와 h를 각각 20픽셀씩 크게 만듭니다.
        w += 20
        h += 40

        # 바운딩 박스가 이미지 경계를 벗어나지 않도록 조정
        x = max(0, x)
        y = max(0, y)
        w = min(image_array.shape[1] - x, w)
        h = min(image_array.shape[0] - y, h)

        # 마스크 이미지에 하얀색으로 채우기
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)
        
    return mask