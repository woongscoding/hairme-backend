FROM public.ecr.aws/lambda/python:3.12

# 의존성 설치 (재현 가능한 빌드를 위해 버전 고정)
# - torch 2.5.1 (CPU): 서빙 Lambda(torch 2.1.0)에서 체크포인트를 로드할 수 있는 조합
# - numpy 1.x: torch/서빙과의 ABI 호환 (numpy 2.x 금지)
# - requirements.txt는 범위 지정(>=)이라 빌드 재현성이 없으므로 사용하지 않는다.
#   의존성은 아래 pip install에 명시된 핀이 단일 소스이다.
RUN pip install --no-cache-dir "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "numpy>=1.26,<2.0" "boto3>=1.28.0,<2.0"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
