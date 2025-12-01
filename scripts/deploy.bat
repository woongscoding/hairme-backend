@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo AWS Lambda Deployment (Windows) - v3 (Fix Manifest)
echo ==========================================

REM Default Configuration
set REGION=ap-northeast-2
set REPO_NAME=hairme-lambda
set FUNCTION_NAME=hairme-analyze
set MEMORY_SIZE=1536
set ROLE_NAME=hairme-lambda-role

REM Check AWS CLI
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] AWS CLI is not installed or not in PATH.
    exit /b 1
)

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    exit /b 1
)

REM Get Account ID
echo [INFO] Getting AWS Account ID...
for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set ACCOUNT_ID=%%i

if "%ACCOUNT_ID%"=="" (
    echo [ERROR] Failed to get AWS Account ID. Please check your credentials.
    exit /b 1
)

echo [INFO] Account ID: %ACCOUNT_ID%
echo [INFO] Region: %REGION%

set ECR_REGISTRY=%ACCOUNT_ID%.dkr.ecr.%REGION%.amazonaws.com
set IMAGE_URI=%ECR_REGISTRY%/%REPO_NAME%:latest

REM Login to ECR
echo [INFO] Logging in to ECR...
aws ecr get-login-password --region %REGION% | docker login --username AWS --password-stdin %ECR_REGISTRY%
if %errorlevel% neq 0 exit /b %errorlevel%

REM Build Docker Image
echo [INFO] Building Docker image (x86_64)...
docker build -f Dockerfile.lambda -t %REPO_NAME%:latest -t %IMAGE_URI% --platform linux/amd64 --provenance=false .
if %errorlevel% neq 0 exit /b %errorlevel%

REM Push to ECR
echo [INFO] Pushing image to ECR...
docker push %IMAGE_URI%
if %errorlevel% neq 0 exit /b %errorlevel%

REM Check if Lambda function exists
echo [INFO] Checking if Lambda function exists...
aws lambda get-function --function-name %FUNCTION_NAME% --region %REGION% >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Function not found. Preparing to create...

    REM Check/Create IAM Role
    echo [INFO] Checking IAM Role...
    aws iam get-role --role-name %ROLE_NAME% >nul 2>&1
    if !errorlevel! neq 0 (
        echo [INFO] Creating IAM Role %ROLE_NAME%...
        
        REM Create Trust Policy
        echo {"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": {"Service": "lambda.amazonaws.com"},"Action": "sts:AssumeRole"}]} > trust-policy.json
        
        aws iam create-role --role-name %ROLE_NAME% --assume-role-policy-document file://trust-policy.json
        del trust-policy.json
        
        echo [INFO] Attaching policies...
        aws iam attach-role-policy --role-name %ROLE_NAME% --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        aws iam attach-role-policy --role-name %ROLE_NAME% --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
        
        echo [INFO] Waiting for role propagation...
        timeout /t 10 /nobreak >nul
    )
    
    REM Get Role ARN
    for /f "tokens=*" %%i in ('aws iam get-role --role-name %ROLE_NAME% --query Role.Arn --output text') do set ROLE_ARN=%%i
    
    echo [INFO] Creating Lambda function...
    aws lambda create-function ^
        --function-name %FUNCTION_NAME% ^
        --package-type Image ^
        --code ImageUri=%IMAGE_URI% ^
        --role !ROLE_ARN! ^
        --memory-size %MEMORY_SIZE% ^
        --timeout 30 ^
        --environment "Variables={USE_DYNAMODB=true,DYNAMODB_TABLE_NAME=hairme-analysis,LOG_LEVEL=INFO,MODEL_NAME=gemini-1.5-flash-latest}" ^
        --architectures x86_64 ^
        --region %REGION%
        
    if !errorlevel! neq 0 exit /b !errorlevel!
    echo [SUCCESS] Function created successfully!

) else (
    echo [INFO] Function exists. Updating...
    
    REM Update Lambda Code
    echo [INFO] Updating Lambda function code...
    aws lambda update-function-code --function-name %FUNCTION_NAME% --image-uri %IMAGE_URI% --region %REGION%
    if !errorlevel! neq 0 exit /b !errorlevel!

    REM Update Lambda Configuration
    echo [INFO] Updating Lambda configuration...
    aws lambda update-function-configuration ^
        --function-name %FUNCTION_NAME% ^
        --memory-size %MEMORY_SIZE% ^
        --environment "Variables={USE_DYNAMODB=true,DYNAMODB_TABLE_NAME=hairme-analysis,LOG_LEVEL=INFO,MODEL_NAME=gemini-1.5-flash-latest}" ^
        --region %REGION%
    if !errorlevel! neq 0 exit /b !errorlevel!
    
    echo [SUCCESS] Function updated successfully!
)

echo.
echo [SUCCESS] Deployment completed successfully!
pause
