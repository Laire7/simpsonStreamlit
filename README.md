# simpsonStreamlit
심손 얼굴 탐지기 Streamlit 서비스

![image (3)](https://github.com/user-attachments/assets/d82465a0-f671-4f45-8871-931a48fedb6f)

## YOLO 모델 학습
[모델 연구 발표: https://docs.google.com/presentation/d/1TlvXmwgOwYb0RXJ3tzjqQJr55Klr2mMIZnCeW2wSWxk/edit?usp=sharing](https://docs.google.com/presentation/d/1TlvXmwgOwYb0RXJ3tzjqQJr55Klr2mMIZnCeW2wSWxk/edit?usp=sharing)
* 기존 데이터 학습한 모델 VS Albumentation을 사용해 Maggie 데이터를 추가한 모델
  * 오히려 가공으로 추가한 모델의 선능이 더 떨어지는 것으로 나왔다 
* 데이터를 가공해서 추가한 모델 VS 수는 적어도 순 데이터를 늘려서 학습한 모델
  * 데이터의 수가 더 가공 한 데이터에 비해 2/3 밖에 되지 않았는데도, 순 데이터를 추가한 모델에서 확연히 좋은 결과가 나왔다
  * 데이터의 수보다, 데이터의 질이 더 중요하다는 것으로 들어났다 
* 기존 데이터 학습한 모델 VS Maggie 데이터를 추가한 모델
  * Maggie 데이터를 100개 이상만 추가해도 선능 개선이 확연히 도드라지고, 데이터가 많은 다른 캐릭터들에도 비교해도 선능이 거의 일치하게 나왔다 
* Yolo Nano (작은 모델) VS Yolo Small (큰 모델) 비교
  * Yolo Small에서 overfitting 현상이 보이면서, Yolo Nano보다 선능이 떨어졌다
  * 캐릭터 그림이 단순해서 이런 형상이 이러난 것으로 추출   
  * 모델의 크기보다 데이터에 맞는 모델을 고르는게 더 중요하다 
     
## 샘플 이미지 검출
![image (2)](https://github.com/user-attachments/assets/8c6a81f1-f11b-4267-bd2e-9a6258670a6e)

## 업로드 된 이미지 검출
![image (4)](https://github.com/user-attachments/assets/8b714390-3f6f-4c2d-b8e7-8e1a28b9561f)
