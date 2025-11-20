from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (프론트와 연결 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 색각친화 팔레트 (대표 6색)
safe_colors = np.array([
    [0, 114, 178],   # 진한 파랑
    [230, 159, 0],   # 주황
    [86, 180, 233],  # 하늘색
    [240, 228, 66],  # 밝은 노랑
    [213, 94, 0],    # 오렌지/레드 계열
    [204, 121, 167]  # 보라
], dtype=np.uint8)

@app.post("/convert")
async def convert(file: UploadFile):
    # 이미지 읽기
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. 색 군집 분해 (k-means)
    h, w, c = img.shape
    flat = img.reshape(-1, 3)

    k = 6
    kmeans = KMeans(n_clusters=k, random_state=0).fit(flat)
    labels = kmeans.labels_

    # 2. 군집 색 → 색각친화 팔레트로 매핑
    new_colors = safe_colors[:k]
    recolored = new_colors[labels].reshape(h, w, 3)

    # 3. JPEG로 다시 변환해 반환
    _, encoded = cv2.imencode(".jpg", recolored)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
