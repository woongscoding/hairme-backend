"""
학습 데이터의 점수 분포를 개선하는 스크립트

문제: 40~70점 범위가 비어있어서 모델이 중간 점수를 예측하지 못함
해결: 기존 데이터에 노이즈를 추가하여 중간 점수 생성
"""

import numpy as np
from pathlib import Path

print("="*60)
print("학습 데이터 점수 분포 수정")
print("="*60)

# 원본 데이터 로드
data = np.load('data_source/ai_face_1000.npz', allow_pickle=True)

face_features = data['face_features']
skin_features = data['skin_features']
hairstyles = data['hairstyles']
scores = data['scores']

print(f"\n[원본] 샘플 수: {len(scores):,}개")
print(f"[원본] 점수 범위: {scores.min():.1f} ~ {scores.max():.1f}")
print(f"[원본] 평균: {scores.mean():.1f}, 표준편차: {scores.std():.1f}")

# 점수대별 카운트
bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
hist, _ = np.histogram(scores, bins=bins)
print(f"\n[원본] 점수 분포:")
for i in range(len(bins) - 1):
    print(f"  {bins[i]:3d}~{bins[i+1]:3d}점: {hist[i]:5d}개 ({hist[i]/len(scores)*100:5.1f}%)")

# 중간 점수 데이터 생성 전략:
# 1. 70~80점 데이터의 복사본을 만들고 점수를 40~70 범위로 조정
# 2. 0~30점 데이터의 복사본을 만들고 점수를 40~70 범위로 조정

# 70~80점 데이터 찾기
high_mask = (scores >= 70) & (scores < 80)
high_indices = np.where(high_mask)[0]

# 0~30점 데이터 찾기
low_mask = scores < 30
low_indices = np.where(low_mask)[0]

# 중간 점수 생성 (각 범위에서 일부 샘플링)
n_mid_samples = 1000  # 중간 점수 샘플 개수

# 70~80점 데이터에서 샘플링 → 40~70점으로 변환
n_from_high = n_mid_samples // 2
sampled_high = np.random.choice(high_indices, size=n_from_high, replace=True)

# 0~30점 데이터에서 샘플링 → 40~70점으로 변환
n_from_low = n_mid_samples - n_from_high
sampled_low = np.random.choice(low_indices, size=n_from_low, replace=True)

# 새로운 중간 점수 데이터 생성
new_face_features = []
new_skin_features = []
new_hairstyles = []
new_scores = []

# 70~80점 → 55~70점으로 변환
for idx in sampled_high:
    new_face_features.append(face_features[idx])
    new_skin_features.append(skin_features[idx])
    new_hairstyles.append(hairstyles[idx])
    # 원래 점수에서 15~25점을 뺌
    new_score = scores[idx] - np.random.uniform(15, 25)
    new_scores.append(max(55, min(70, new_score)))

# 0~30점 → 40~55점으로 변환
for idx in sampled_low:
    new_face_features.append(face_features[idx])
    new_skin_features.append(skin_features[idx])
    new_hairstyles.append(hairstyles[idx])
    # 원래 점수에 20~35점을 더함
    new_score = scores[idx] + np.random.uniform(20, 35)
    new_scores.append(max(40, min(55, new_score)))

# 원본 데이터와 병합
combined_face_features = np.vstack([face_features, np.array(new_face_features)])
combined_skin_features = np.vstack([skin_features, np.array(new_skin_features)])
combined_hairstyles = np.concatenate([hairstyles, np.array(new_hairstyles)])
combined_scores = np.concatenate([scores, np.array(new_scores)])

print(f"\n[수정] 샘플 수: {len(combined_scores):,}개 (+{len(new_scores):,})")
print(f"[수정] 점수 범위: {combined_scores.min():.1f} ~ {combined_scores.max():.1f}")
print(f"[수정] 평균: {combined_scores.mean():.1f}, 표준편차: {combined_scores.std():.1f}")

# 수정된 점수대별 카운트
hist_new, _ = np.histogram(combined_scores, bins=bins)
print(f"\n[수정] 점수 분포:")
for i in range(len(bins) - 1):
    print(f"  {bins[i]:3d}~{bins[i+1]:3d}점: {hist_new[i]:5d}개 ({hist_new[i]/len(combined_scores)*100:5.1f}%)")

# 새로운 파일로 저장
output_path = 'data_source/ai_face_1000_fixed.npz'
np.savez(
    output_path,
    face_features=combined_face_features,
    skin_features=combined_skin_features,
    hairstyles=combined_hairstyles,
    scores=combined_scores
)

print(f"\n[저장] 수정된 데이터 저장: {output_path}")
print("\n" + "="*60)
print("완료! 이제 이 파일로 모델을 재학습하세요:")
print(f"  python scripts/train_model_v4_no_leakage.py --data {output_path}")
print("="*60)
