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
    
    for (center_x, center_y, width, height) in boxes:
        # bounding box의 좌표 계산
        x = int((center_x - width / 2) * image_array.shape[1])
        y = int((center_y - height / 2) * image_array.shape[0])
        w = int(width * image_array.shape[1])
        h = int(height * image_array.shape[0])
        
        # x와 y를 각각 5픽셀씩 작게 만듭니다.
        x -= 10
        y -= 20

        # w와 h를 각각 10픽셀씩 크게 만듭니다.
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

# frame 단위 이미지 저장
def frame_save(video_path):
    # 이미지를 저장할 디렉토리 경로
    # frame_directory = 'frame_save/'

    static_folder = 'media/'
    frame_video_path = os.path.join(static_folder, 'frame_save')

    if not os.path.exists(frame_video_path):
        os.makedirs(frame_video_path)

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 프레임 간격
    frame_interval = 10

    # 프레임 수 초기화
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 각 프레임을 이미지로 저장
        frame_count += 1
        if frame_count % frame_interval == 0:
            image_filename = os.path.join(frame_video_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(image_filename, frame)

    cap.release()

    return frame_video_path

# 이미지 영상으로 변환
def set_video(results_inference_video_path, target_video):
    # 이미지 파일들이 있는 폴더 경로
    image_folder = results_inference_video_path
    
    target_video = target_video.replace('media/videos/', '')

    # 결과 MP4 파일 경로 및 설정
    static_folder = 'media/'
    output_folder = os.path.join(static_folder, 'results_video')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 생성된 영상 이름 input 된 파일 이름
    output_video_path = os.path.join(output_folder, target_video)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 코덱 설정
    frame_rate = 2.0

    # 이미지 파일 목록 가져오기
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 이미지 파일들을 정렬
    image_files.sort()

    # MP4 비디오 파일 생성
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 이미지를 비디오에 추가
    for image_file in image_files:
        video.write(cv2.imread(image_file))
        
    video.release()

# 영상 생성 후 frame 이미지 / 추론 이미지 폴더 내용 삭제
def delete_folder_contents(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
