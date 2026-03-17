## 프로젝트 개요 

- 기존 로드 뷰 서비스의 데이터 수집 중 일어나는 사생활 침해 문제 인식
- 객체 탐지 모델을 활용하여 사람을 인식한다
- 인식된 사람을 지운 상태로 로드뷰 데이터를 수집하는 것을 목표로 한다

## 주요 기능

- 로드뷰 데이터를 수집 형태와 같이 영상 또는 이미지 형태의 source에 대해 동작
- YOLO 모델로 영상 또는 이미지내에 있는 사람(person) 객체를 Object detection
- polygon 형태로 객체의 형상에 맞는 픽셀 좌표 정보를 받아 Mask 생성
- Mask를 기반으로 원본 이미지에 LaMa 모델을 활용한 inpainting(Erase)
- 단순히 Mask를 덧대어 가리는 수준이 아닌 뒤의 배경과 자연스레 이어지는 이미지 생성


## ⚙️ 환경 설정
- pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
- Python 3.11
