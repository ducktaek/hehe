from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
# from webex_utils import webex_router as webex_router

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# CORS(Cross-Origin Resource Sharing) 설정
# 모든 출처(origin)에서의 접근을 허용함
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # 모든 출처 허용
    allow_credentials=True,          # 쿠키 포함 허용
    allow_methods=["*"],             # 모든 HTTP 메서드 허용
    allow_headers=["*"],             # 모든 헤더 허용
)

# /predict 엔드포인트: 클라이언트가 이미지 파일을 POST하면 객체 감지 결과 반환
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 업로드된 이미지 파일을 읽음 (바이트 형식)
    contents = await file.read()

    # 바이트 배열을 numpy 배열로 변환
    nparr = np.frombuffer(contents, np.uint8)

    # OpenCV를 이용해 이미지를 디코딩 (BGR 형식)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ※ 이 부분에서 YOLO 모델 등을 활용해 실제 객체 감지를 수행하면 됨
    # 현재는 예시 결과만 반환
    predictions = [
        {"class": "person", "confidence": 0.88},
        {"class": "dog", "confidence": 0.76}
    ]

    # JSON 형식으로 감지 결과를 응답
    return JSONResponse(content={"predictions": predictions})

