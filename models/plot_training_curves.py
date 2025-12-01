import json
import matplotlib.pyplot as plt
import os

# 현재 스크립트 위치 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(script_dir, 'training_history_v5_normalized.json')

with open(history_path, 'r', encoding='utf-8') as f:
    history = json.load(f)

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss Curve
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2196F3')
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#FF5722')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('학습 손실 곡선 (Loss Curve)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# MAE Curve (원본 스케일 MAE 사용)
axes[1].plot(history['val_mae_original'], label='Validation MAE', linewidth=2, color='#9C27B0')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE (점)', fontsize=12)
axes[1].set_title('평균 절대 오차 곡선 (MAE Curve)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 최종 성능 표시
final_val_mae = history['val_mae_original'][-1]
axes[1].axhline(y=final_val_mae, color='#9C27B0', linestyle='--', alpha=0.5)
axes[1].annotate(f'Final Val MAE: {final_val_mae:.2f}점',
                 xy=(len(history['val_mae_original'])-1, final_val_mae),
                 xytext=(len(history['val_mae_original'])*0.6, final_val_mae + 2),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()

# 저장
output_path = os.path.join(script_dir, 'training_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"그래프 저장 완료: {output_path}")

# 성능 요약 출력
print("\n" + "="*50)
print("모델 성능 요약")
print("="*50)
print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
print(f"Final Val MAE: {final_val_mae:.2f}점")
print(f"점수 범위: 10~95점 (85점 범위)")
print(f"오차율: {(final_val_mae / 85) * 100:.1f}%")
print("="*50)

plt.show()