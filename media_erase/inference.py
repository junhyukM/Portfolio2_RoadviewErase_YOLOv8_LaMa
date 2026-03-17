from . model_load import load_lama_cleaner, load_yolo
from . util import get_mask, norm_img, set_video, frame_save, delete_folder_contents
from PIL import Image
import cv2
import numpy as np
import torch
import os

# 변수에 이미지를 받아서 yolo 추론에 넣어야함
def yolo_inference(image_path):
    # yolo 추론
    model = load_yolo()
    # 이미지 추론시 size 조정이 필요함
    results = model(image_path)
    
    boxes = results[0].boxes
    
    xywhn = []
    # class가 '0' 인 bounding box 좌표 리스트에 저장
    
    for i in range(len(boxes)):
        h_class = boxes.cls[i].cpu().numpy()
        if h_class == 0.0:
            xywhn.append(boxes[i].xywhn.cpu().numpy().tolist())
            
    # 3차원 list -> 2차원 list 변환
    flattened_2d_list = [item for sublist in xywhn for item in sublist]
    
    # 2차원 list -> 소수점 6자리 까지 반올림
    rounded_list = [[round(val, 6) for val in sublist] for sublist in flattened_2d_list]
    
    # 최종 좌표 초기화
    final_xywhn = []

    # 최종 좌표 list로 저장
    for result in rounded_list:
        final_xywhn.append(result)

    # print(final_xywhn)
    
    return final_xywhn

def lama_cleaner(image: np.ndarray, mask: np.ndarray, device: str):
    model = load_lama_cleaner()
    
    image = norm_img(image)
    mask = norm_img(mask)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    inpainted_image = model(image, mask) # inference

    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")

    return Image.fromarray(cur_res)

# 영상 경로 가져와야함
def video_inference(target_video):
    # video_path = 'input_media/video.mp4'
    static_folder = 'media/'

    # 추론 결과 이미지 폴더가 없으면 생성
    results_inference_video_path = os.path.join(static_folder, 'results_inference_videos')
    
    if not os.path.exists(results_inference_video_path):
        os.makedirs(results_inference_video_path)

    # 추론할 이미지
    folder_path = frame_save(target_video)

    # 폴더 내의 모든 파일 목록 가져오기
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
    
        # 이미지 로드
        image = Image.open(image_path)
    
        # image 파일이 PNG인 경우 채널 에러가 날 수 있으므로 RGB로 변환
        image = image.convert('RGB')
        
        # image resize
        original_width, original_height = image.size
        new_width = original_width
        new_height = original_height

        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32

        image = image.resize((new_width, new_height))

        image_array = np.array(image)

        # yolo 추론
        boxes = yolo_inference(image_path)

        # box 변환 후 마스크 get
        get_mask_image = get_mask(boxes, image_array)

        # lama 추론
        yolo_lama_cleaner = lama_cleaner(image_array, get_mask_image, device='cpu')

        # 추론 이미지 경로
        result_path = f'media/results_inference_videos/{image_path.split("/")[-1]}'
        result_path = result_path.replace("/frame_save", "")

        # 추론 이미지 저장
        yolo_lama_cleaner.save(result_path)

    set_video(results_inference_video_path, target_video)

    delete_folder_contents('media/frame_save')
    delete_folder_contents('media/results_inference_videos')


if __name__ == "__main__":

    # TODO: 이미지, 마스크 변수 초기화 후 실행
    # image = np.array()
    # mask = np.array()
    # device='cuda'
    
    # lama_cleaner(image=image, mask=mask, device=device)
    # video_inference()
    pass
    