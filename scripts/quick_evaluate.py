#!/usr/bin/env python3
"""간단한 v4 모델 평가 스크립트"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from models.ml_recommender_v4 import ContinuousRecommenderV4
from sentence_transformers import SentenceTransformer

print('='*60)
print('Model Evaluation Starting...')
print('='*60)

# 모델 로드
device = torch.device('cpu')
model = ContinuousRecommenderV4().to(device)
checkpoint = torch.load('models/hairstyle_recommender_v4.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Model loaded successfully')

# 데이터 로드
data = np.load('data_source/ai_face_1000.npz', allow_pickle=True)
print(f'Data loaded: {len(data["scores"]):,} samples')

# 헤어스타일 임베딩
print('Generating embeddings...')
sent_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
style_embs = sent_model.encode(data['hairstyles'].tolist(), show_progress_bar=False)

# 텐서 변환
face_feat = torch.tensor(data['face_features'], dtype=torch.float32).to(device)
skin_feat = torch.tensor(data['skin_features'], dtype=torch.float32).to(device)
style_emb = torch.tensor(style_embs, dtype=torch.float32).to(device)
scores_true = data['scores']

# 예측
print('Predicting...')
with torch.no_grad():
    preds = model(face_feat, skin_feat, style_emb)
    preds_np = preds.cpu().numpy().flatten()

# 평가
mae = np.abs(preds_np - scores_true).mean()
mse = ((preds_np - scores_true) ** 2).mean()
rmse = np.sqrt(mse)
corr = np.corrcoef(preds_np, scores_true)[0, 1]

print()
print('='*60)
print('Overall Performance:')
print(f'  MAE:  {mae:.2f}')
print(f'  RMSE: {rmse:.2f}')
print(f'  Correlation: {corr:.3f}')
print('='*60)
print('Evaluation completed successfully!')
print('='*60)
