from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 색각친화 팔레트
safe_colors = np.array([
    [0, 114, 178],   # 파랑
    [230, 159, 0],   # 주황
    [86, 180, 233],  # 하늘색
    [240, 228, 66],  # 밝은 노랑
    [213, 94, 0],    # 오렌지/레드
    [204, 121, 167]  # 보라
], dtype=np.uint8)


@app.post("/convert")
async def convert(file: UploadFile):

    # -----------------------------------------
    # 1. 이미지 안전하게 읽기 (Render/Cloud 최적화 방식)
    # -----------------------------------------
    img_bytes = await file.read()

    # 임시 파일로 저장하여 OpenCV가 안정적으로 읽도록 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)

    if img is None:
        return {"error": "이미지 디코딩 실패 - 서버에서 파일이 손상되어 읽기 불가능합니다."}

    # -----------------------------------------
    # 2. 색 군집화 (k-means)
    # -----------------------------------------
    h, w, c = img.shape
    flat = img.reshape(-1, 3)

    k = 6
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(flat)

    # -----------------------------------------
    # 3. 색각친화 팔레트로 재색상화
    # -----------------------------------------
    new_colors = safe_colors[:k]
    recolored = new_colors[labels].reshape(h, w, 3)

    # -----------------------------------------
    # 4. JPEG로 인코딩 후 반환
    # -----------------------------------------
    _, encoded = cv2.imencode(".jpg", recolored)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
