FROM public.ecr.aws/lambda/python:3.12

# 의존성 설치 - requirements.txt 가 단일 소스다 (버전을 여기에 다시 적지 않는다).
# torch CPU 인덱스와 버전 고정 사유는 requirements.txt 주석 참조.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
