import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 모델 로드
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name + ".pt")

HOME = os.getcwd()
# 업로드된 이미지를 저장할 디렉터리 설정
UPLOAD_DIR = os.path.join(HOME, "uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 파일 업로드
st.title("이미지 업로드 및 YOLOv8 객체 탐지")

# Sidebar에서 YOLO 모델 선택
st.sidebar.title("YOLO 모델 선택")
model_type = st.sidebar.radio("모델을 선택하세요", ("yolov8n", "yolov8s", "yolov8m"))
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 원본 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    imageBGR = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #image = cv2.resize(imageBGR, (640, 640))

    # 모델 로드 및 예측
    model = load_model(model_type)
    #results = model(imageBGR)
    results = model.predict(source=imageBGR, conf=confidence)

    # 결과 이미지 저장
    result_img = results[0].plot()  # 결과 플롯 생성
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    result_image = Image.fromarray(result_img)  # 배열을 PIL 이미지로 변환
    # 결과 출력
    col1, col2 = st.columns(2)
    with col1:
        st.header("입력된 이미지")
        st.image(image, use_column_width=True)
    
    with col2:
        st.header("탐지 결과")
        st.image(result_image, use_column_width=True)