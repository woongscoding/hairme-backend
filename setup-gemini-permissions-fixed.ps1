# ============================================
# ECS Task Execution Role에 Gemini Secret 접근 권한 추가
# JSON 파일 사용 버전
# ============================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Gemini Secret 접근 권한 설정" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: 현재 Role 정책 확인
Write-Host "Step 1: 현재 Role 정책 확인" -ForegroundColor Yellow
aws iam list-attached-role-policies `
    --role-name ecsTaskExecutionRole-Hairme `
    --region ap-northeast-2

Write-Host ""

# Step 2: JSON 파일로 정책 추가
Write-Host "Step 2: Gemini Secret 접근 권한 추가 (JSON 파일 사용)" -ForegroundColor Yellow

# gemini-secret-policy.json 파일이 현재 디렉토리에 있어야 합니다
if (!(Test-Path "gemini-secret-policy.json")) {
    Write-Host "❌ 오류: gemini-secret-policy.json 파일을 찾을 수 없습니다!" -ForegroundColor Red
    Write-Host "이 파일을 현재 디렉토리에 생성해주세요." -ForegroundColor Yellow
    exit 1
}

aws iam put-role-policy `
    --role-name ecsTaskExecutionRole-Hairme `
    --policy-name GeminiSecretAccess `
    --policy-document file://gemini-secret-policy.json `
    --region ap-northeast-2

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 권한 추가 성공!" -ForegroundColor Green
} else {
    Write-Host "❌ 권한 추가 실패!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: 권한 추가 확인
Write-Host "Step 3: 권한 추가 확인" -ForegroundColor Yellow
aws iam get-role-policy `
    --role-name ecsTaskExecutionRole-Hairme `
    --policy-name GeminiSecretAccess `
    --region ap-northeast-2

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ 완료!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "이제 ECS Task가 Gemini API 키를 읽을 수 있습니다." -ForegroundColor Cyan
Write-Host "다음 명령어로 배포를 진행하세요:" -ForegroundColor Yellow
Write-Host "  .\deploy-phase1.ps1" -ForegroundColor Cyan
Write-Host ""