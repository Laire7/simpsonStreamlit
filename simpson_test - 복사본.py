import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import glob
import time

# 모델 로드
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name + ".pt")

# 파일 업로드
st.title("Simpson Yolov11 객체 탐지")

# Sidebar에서 YOLO 모델 선택
st.sidebar.title("YOLO 모델 선택")
model_type = st.sidebar.radio("모델을 선택하세요", ("simpson_orig", "simpson_albu", "simpson_addOrig"))
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
data_type = st.sidebar.radio("데이터셋을 선택하세요", ("sample_images", "custom"))
# data_type = st.sidebar.radio("데이터셋을 선택하세요", ("sample_images", "custom", "sample_video"))
uploaded_file = None
if data_type == "custom":   
  uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

def infer_image(img_file, model_type, imageBGR):
   # 모델 로드 및 예측
    model = load_model(model_type)
    results = model(imageBGR)
    results = model.predict(source=imageBGR, conf=confidence)

    # 결과 이미지 저장
    result_img = results[0].plot()  # 결과 플롯 생성
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    result_image = Image.fromarray(result_img)  # 배열을 PIL 이미지로 변환
    # 결과 출력
    return result_image
  
def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/simpson/test_data/simpson_video.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()
  
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
          
sample_img_file = None
if data_type == 'sample_images':
    # get all sample images
    img_path = glob.glob('simpson/sample_images/*')
    img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
    sample_img_file = img_path[img_slider - 1]
      
elif data_type == 'sample_video':
      # get all sample images
    vid_path = glob.glob('simpson/sample_videos/*')
    vid_slider = st.slider("Select a test video.", min_value=1, max_value=len(vid_path), step=1)
    sample_vid_file = vid_path[vid_slider - 1]  
    with st.spinner("Predicting on video..."):
      temp_video_path = os.path.join(vid_path, "uploaded_video.mp4")
      with open(temp_video_path, "wb") as f:
          f.write(uploaded_file.read())
      
      results = model.predict(source=temp_video_path, conf=confidence, save=True)

    # 결과 동영상 표시
    result_video_path = os.path.join(HOME, results[0].save_dir, "uploaded_video.avi")
    st.video(result_video_path, format="video/avi")
else:
  st.error("선택한 입력 유형과 업로드한 파일이 일치하지 않습니다.")
  
if sample_img_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_img_file, caption="Selected Image")
    with col2:
      image = Image.open(sample_img_file)
      imageBGR = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
      img = infer_image(sample_img_file, model_type, imageBGR)
      st.image(img, caption="Model prediction")
else:
  HOME = os.getcwd()
  # 업로드된 이미지를 저장할 디렉터리 설정
  UPLOAD_DIR = os.path.join(HOME, "uploads")
  if not os.path.exists(UPLOAD_DIR):
      os.makedirs(UPLOAD_DIR)