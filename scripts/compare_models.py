#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""두 모델 비교: 기존 vs 리키지 방지"""

import sys
import io
from pathlib import Path

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from models.ml_recommender_v4 import ContinuousRecommenderV4
from sentence_transformers import SentenceTransformer

print('='*80)
print('모델 비교 평가: 기존 vs 리키지 방지')
print('='*80)

# 데이터 로드
data = np.load('data_source/ai_face_1000.npz', allow_pickle=True)
print(f'\n데이터: {len(data["scores"]):,}개 샘플')

# 헤어스타일 임베딩
sent_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
style_embs = sent_model.encode(data['hairstyles'].tolist(), show_progress_bar=False)

# 텐서 변환
device = torch.device('cpu')
face_feat = torch.tensor(data['face_features'], dtype=torch.float32).to(device)
skin_feat = torch.tensor(data['skin_features'], dtype=torch.float32).to(device)
style_emb = torch.tensor(style_embs, dtype=torch.float32).to(device)
scores_true = data['scores']

# 모델 1: 기존 (리키지 있음)
print('\n' + '='*80)
print('📊 모델 1: 기존 (리키지 가능성 있음)')
print('='*80)
model1 = ContinuousRecommenderV4().to(device)
checkpoint1 = torch.load('models/hairstyle_recommender_v4.pt', map_location=device, weights_only=False)
model1.load_state_dict(checkpoint1['model_state_dict'])
model1.eval()

with torch.no_grad():
    preds1 = model1(face_feat, skin_feat, style_emb).cpu().numpy().flatten()

mae1 = np.abs(preds1 - scores_true).mean()
mse1 = ((preds1 - scores_true) ** 2).mean()
rmse1 = np.sqrt(mse1)
corr1 = np.corrcoef(preds1, scores_true)[0, 1]
r2_1 = 1 - (np.sum((scores_true - preds1) ** 2) / np.sum((scores_true - scores_true.mean()) ** 2))

print(f'  MAE:         {mae1:.2f}')
print(f'  RMSE:        {rmse1:.2f}')
print(f'  Correlation: {corr1:.4f}')
print(f'  R²:          {r2_1:.4f}')
print(f'  Best epoch:  {checkpoint1["epoch"]}')

# 모델 2: 리키지 방지
print('\n' + '='*80)
print('📊 모델 2: 리키지 방지 (얼굴 단위 분할)')
print('='*80)
model2 = ContinuousRecommenderV4().to(device)
checkpoint2 = torch.load('models/hairstyle_recommender_v4_no_leakage.pt', map_location=device, weights_only=False)
model2.load_state_dict(checkpoint2['model_state_dict'])
model2.eval()

with torch.no_grad():
    preds2 = model2(face_feat, skin_feat, style_emb).cpu().numpy().flatten()

mae2 = np.abs(preds2 - scores_true).mean()
mse2 = ((preds2 - scores_true) ** 2).mean()
rmse2 = np.sqrt(mse2)
corr2 = np.corrcoef(preds2, scores_true)[0, 1]
r2_2 = 1 - (np.sum((scores_true - preds2) ** 2) / np.sum((scores_true - scores_true.mean()) ** 2))

print(f'  MAE:         {mae2:.2f}')
print(f'  RMSE:        {rmse2:.2f}')
print(f'  Correlation: {corr2:.4f}')
print(f'  R²:          {r2_2:.4f}')
print(f'  Best epoch:  {checkpoint2["epoch"]}')

# 비교
print('\n' + '='*80)
print('📈 성능 비교')
print('='*80)
print(f'{"지표":<15} {"기존":>12} {"리키지방지":>12} {"차이":>12} {"개선":>8}')
print('-'*80)

def compare(name, val1, val2, lower_is_better=True):
    diff = val2 - val1
    if lower_is_better:
        improved = '✅' if diff < 0 else '❌'
        pct = -diff / val1 * 100
    else:
        improved = '✅' if diff > 0 else '❌'
        pct = diff / val1 * 100
    print(f'{name:<15} {val1:>12.4f} {val2:>12.4f} {diff:>+12.4f} {improved} {pct:>+6.2f}%')

compare('MAE', mae1, mae2, lower_is_better=True)
compare('RMSE', rmse1, rmse2, lower_is_better=True)
compare('Correlation', corr1, corr2, lower_is_better=False)
compare('R²', r2_1, r2_2, lower_is_better=False)

# 예측 차이 분석
print('\n' + '='*80)
print('🔍 예측 차이 분석')
print('='*80)
pred_diff = np.abs(preds1 - preds2)
print(f'  평균 예측 차이:  {pred_diff.mean():.2f}')
print(f'  최대 예측 차이:  {pred_diff.max():.2f}')
print(f'  차이 > 5점:     {(pred_diff > 5).sum():,}개 ({(pred_diff > 5).sum()/len(pred_diff)*100:.1f}%)')
print(f'  차이 > 10점:    {(pred_diff > 10).sum():,}개 ({(pred_diff > 10).sum()/len(pred_diff)*100:.1f}%)')

# 점수 구간별 비교
print('\n' + '='*80)
print('📊 점수 구간별 MAE 비교')
print('='*80)
bins = [(0, 40), (40, 60), (60, 80), (80, 100)]
print(f'{"구간":<12} {"기존 MAE":>12} {"방지 MAE":>12} {"샘플수":>10}')
print('-'*80)

for low, high in bins:
    mask = (scores_true >= low) & (scores_true < high)
    if mask.sum() > 0:
        mae1_bin = np.abs(preds1[mask] - scores_true[mask]).mean()
        mae2_bin = np.abs(preds2[mask] - scores_true[mask]).mean()
        print(f'{low:3d}-{high:3d} {mae1_bin:>12.2f} {mae2_bin:>12.2f} {mask.sum():>10,}')

print('\n' + '='*80)
print('✅ 비교 완료!')
print('='*80)

# 결론
print('\n🎯 결론:')
if mae2 < mae1:
    print('  ✅ 리키지 방지 모델이 전체적으로 더 우수합니다!')
    print('  ✅ 데이터 리키지가 오히려 성능을 저하시켰을 가능성')
    print('  ✅ 리키지 방지 모델을 프로덕션에 사용하는 것을 권장합니다.')
else:
    print('  ⚠️  기존 모델이 약간 더 좋지만, 리키지로 인한 과대평가 가능성')
    print('  ⚠️  실제 새로운 얼굴에 대해서는 리키지 방지 모델이 더 안정적일 수 있습니다.')
