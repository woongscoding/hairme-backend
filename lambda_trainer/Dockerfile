FROM public.ecr.aws/lambda/python:3.12

# Install dependencies - PyTorch CPU version
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir numpy>=1.24.0 boto3>=1.28.0 && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
