FROM python:3.11-slim

WORKDIR /app

# OpenCV 시스템 의존성 + Haar Cascade 파일 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Haar Cascade 파일 다운로드 (확실하게)
RUN mkdir -p /usr/share/opencv4/haarcascades && \
    wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml \
    -O /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml

# 파일 존재 확인 (빌드 시 검증)
RUN python3 -c "import cv2; import os; \
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'; \
    print(f'Checking: {path}'); \
    assert os.path.exists(path), 'Haar Cascade file not found!'; \
    print('✅ Haar Cascade file exists')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]