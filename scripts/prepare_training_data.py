#!/usr/bin/env python3
"""
ML 학습 데이터 전처리

합성 데이터와 임베딩을 결합하여 ML 모델 학습용 데이터셋 생성

Input features (392차원):
  - 얼굴형 one-hot (4차원)
  - 피부톤 one-hot (4차원)
  - 헤어스타일 임베딩 (384차원)

Target:
  - 추천 점수 (0-100)

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_preprocessor import normalize_style_name


# ==================== 설정 ====================
class Config:
    """스크립트 설정"""

    # 카테고리 옵션 (순서 중요!)
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    # 차원 크기
    FACE_SHAPE_DIM = 4
    SKIN_TONE_DIM = 4
    EMBEDDING_DIM = 384
    TOTAL_INPUT_DIM = FACE_SHAPE_DIM + SKIN_TONE_DIM + EMBEDDING_DIM  # 392

    # Train/Val split
    VAL_RATIO = 0.2
    RANDOM_SEED = 42

    # 입력/출력 경로
    DEFAULT_DATA_PATH = "data_source/final_training_data_3200.json"  # 3855개 조합!
    DEFAULT_EMBEDDING_PATH = "data_source/style_embeddings.npz"
    DEFAULT_OUTPUT_DIR = "data_source"


# ==================== 데이터 로더 ====================
class DataLoader:
    """데이터 및 임베딩 로딩"""

    @staticmethod
    def load_training_data(file_path: Path) -> Dict:
        """학습 데이터 로드"""
        print(f"📂 학습 데이터 로딩: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ✅ 로드 완료: {data['metadata']['total_combinations']}개 조합")
        return data

    @staticmethod
    def load_embeddings(file_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
        """스타일 임베딩 로드"""
        print(f"📂 임베딩 로딩: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        styles = data['styles']
        embeddings = data['embeddings']

        # 스타일명 -> 인덱스 매핑
        style_to_idx = {style: idx for idx, style in enumerate(styles)}

        print(f"  ✅ 로드 완료: {len(styles)}개 스타일, 임베딩 shape {embeddings.shape}")
        return embeddings, style_to_idx


# ==================== 특징 변환 ====================
class FeatureTransformer:
    """특징 벡터 변환"""

    def __init__(self, embeddings: np.ndarray, style_to_idx: Dict[str, int]):
        """
        초기화

        Args:
            embeddings: 스타일 임베딩 배열
            style_to_idx: 스타일명 -> 인덱스 매핑
        """
        self.embeddings = embeddings
        self.style_to_idx = style_to_idx

    @staticmethod
    def encode_face_shape(face_shape: str) -> np.ndarray:
        """얼굴형을 one-hot 인코딩"""
        vec = np.zeros(Config.FACE_SHAPE_DIM, dtype=np.float32)
        if face_shape in Config.FACE_SHAPES:
            idx = Config.FACE_SHAPES.index(face_shape)
            vec[idx] = 1.0
        return vec

    @staticmethod
    def encode_skin_tone(skin_tone: str) -> np.ndarray:
        """피부톤을 one-hot 인코딩"""
        vec = np.zeros(Config.SKIN_TONE_DIM, dtype=np.float32)
        if skin_tone in Config.SKIN_TONES:
            idx = Config.SKIN_TONES.index(skin_tone)
            vec[idx] = 1.0
        return vec

    def encode_hairstyle(self, hairstyle: str) -> np.ndarray:
        """헤어스타일명을 임베딩 벡터로 변환 (띄어쓰기 정규화 적용)"""
        # 정규화 적용
        normalized = normalize_style_name(hairstyle)

        if normalized in self.style_to_idx:
            idx = self.style_to_idx[normalized]
            return self.embeddings[idx].astype(np.float32)
        else:
            # 미등록 스타일은 제로 벡터 (발생하면 안됨)
            print(f"  ⚠️  경고: 미등록 스타일 '{hairstyle}' (정규화: '{normalized}')")
            return np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)

    def transform_combination(self, combo: Dict) -> np.ndarray:
        """
        하나의 조합을 특징 벡터로 변환

        Args:
            combo: 조합 딕셔너리

        Returns:
            특징 벡터 (392차원)
        """
        face_vec = self.encode_face_shape(combo['face_shape'])
        tone_vec = self.encode_skin_tone(combo['skin_tone'])
        style_vec = self.encode_hairstyle(combo['hairstyle'])

        # 연결: [face(4) + tone(4) + style(384)] = 392
        feature = np.concatenate([face_vec, tone_vec, style_vec])

        return feature


# ==================== 데이터셋 생성 ====================
class DatasetBuilder:
    """학습/검증 데이터셋 구축"""

    def __init__(self, transformer: FeatureTransformer):
        """초기화"""
        self.transformer = transformer

    def build_dataset(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        전체 데이터셋 구축

        Args:
            training_data: 학습 데이터 딕셔너리

        Returns:
            (X, y) - 특징 행렬, 타겟 벡터
        """
        print(f"\n🔄 데이터셋 구축 중...")

        X_list = []
        y_list = []

        for image_data in training_data['training_data']:
            for combo in image_data['combinations']:
                # 특징 벡터 생성
                feature = self.transformer.transform_combination(combo)
                X_list.append(feature)

                # 타겟 (추천 점수)
                score = combo['recommendation_score']
                y_list.append(score)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"  ✅ 데이터셋 구축 완료:")
        print(f"    - 샘플 수: {len(X)}개")
        print(f"    - 특징 차원: {X.shape[1]}차원")
        print(f"    - 점수 범위: {y.min():.1f} ~ {y.max():.1f}")
        print(f"    - 점수 평균: {y.mean():.1f} ± {y.std():.1f}")

        return X, y

    @staticmethod
    def split_dataset(
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = Config.VAL_RATIO
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Train/Validation split

        Args:
            X: 특징 행렬
            y: 타겟 벡터
            val_ratio: 검증 데이터 비율

        Returns:
            (X_train, X_val, y_train, y_val)
        """
        print(f"\n📊 Train/Validation Split...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_ratio,
            random_state=Config.RANDOM_SEED,
            shuffle=True
        )

        print(f"  ✅ Split 완료:")
        print(f"    - Train: {len(X_train)}개 ({100*(1-val_ratio):.0f}%)")
        print(f"    - Val:   {len(X_val)}개 ({100*val_ratio:.0f}%)")

        return X_train, X_val, y_train, y_val


# ==================== 저장 ====================
class DatasetExporter:
    """데이터셋 저장"""

    @staticmethod
    def save_dataset(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        output_dir: Path
    ):
        """
        데이터셋을 NPZ 파일로 저장

        Args:
            X_train, X_val, y_train, y_val: 데이터셋
            output_dir: 출력 디렉토리
        """
        print(f"\n💾 데이터셋 저장 중...")

        # NPZ 파일로 저장
        npz_path = output_dir / "ml_training_dataset.npz"
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )

        print(f"  ✅ NPZ 저장: {npz_path}")

        # 메타데이터 저장
        metadata = {
            "feature_dim": Config.TOTAL_INPUT_DIM,
            "face_shape_dim": Config.FACE_SHAPE_DIM,
            "skin_tone_dim": Config.SKIN_TONE_DIM,
            "embedding_dim": Config.EMBEDDING_DIM,
            "face_shapes": Config.FACE_SHAPES,
            "skin_tones": Config.SKIN_TONES,
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "target_min": float(y_train.min()),
            "target_max": float(y_train.max())
        }

        json_path = output_dir / "ml_dataset_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 메타데이터 저장: {json_path}")

        # 파일 크기
        npz_size = npz_path.stat().st_size / 1024
        print(f"\n  📊 파일 크기: {npz_size:.1f} KB")

    @staticmethod
    def save_csv_sample(
        X_train: np.ndarray,
        y_train: np.ndarray,
        output_dir: Path,
        n_samples: int = 100
    ):
        """샘플 CSV 저장 (검증용)"""
        print(f"\n💾 샘플 CSV 저장 중 (처음 {n_samples}개)...")

        # 특징 컬럼명 생성
        feature_cols = []
        feature_cols += [f"face_{shape}" for shape in Config.FACE_SHAPES]
        feature_cols += [f"tone_{tone}" for tone in Config.SKIN_TONES]
        feature_cols += [f"emb_{i}" for i in range(Config.EMBEDDING_DIM)]

        # DataFrame 생성
        df = pd.DataFrame(
            X_train[:n_samples],
            columns=feature_cols
        )
        df['score'] = y_train[:n_samples]

        csv_path = output_dir / "training_sample.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"  ✅ CSV 저장: {csv_path}")


# ==================== 메인 함수 ====================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="ML 학습 데이터 전처리 및 생성"
    )

    parser.add_argument(
        '-d', '--data',
        type=str,
        default=Config.DEFAULT_DATA_PATH,
        help=f'학습 데이터 경로 (기본값: {Config.DEFAULT_DATA_PATH})'
    )

    parser.add_argument(
        '-e', '--embeddings',
        type=str,
        default=Config.DEFAULT_EMBEDDING_PATH,
        help=f'임베딩 파일 경로 (기본값: {Config.DEFAULT_EMBEDDING_PATH})'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=Config.DEFAULT_OUTPUT_DIR,
        help=f'출력 디렉토리 (기본값: {Config.DEFAULT_OUTPUT_DIR})'
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    embedding_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎨 ML 학습 데이터 전처리 v1.0.0")
    print("=" * 60)
    print(f"학습 데이터: {data_path}")
    print(f"임베딩 파일: {embedding_path}")
    print(f"출력 경로: {output_dir.absolute()}")
    print(f"특징 차원: {Config.TOTAL_INPUT_DIM}차원")
    print("=" * 60)

    try:
        # 1. 데이터 로드
        loader = DataLoader()
        training_data = loader.load_training_data(data_path)
        embeddings, style_to_idx = loader.load_embeddings(embedding_path)

        # 2. 특징 변환기 생성
        transformer = FeatureTransformer(embeddings, style_to_idx)

        # 3. 데이터셋 구축
        builder = DatasetBuilder(transformer)
        X, y = builder.build_dataset(training_data)

        # 4. Train/Val split
        X_train, X_val, y_train, y_val = builder.split_dataset(X, y)

        # 5. 저장
        exporter = DatasetExporter()
        exporter.save_dataset(X_train, X_val, y_train, y_val, output_dir)
        exporter.save_csv_sample(X_train, y_train, output_dir)

        print("\n" + "=" * 60)
        print("🎉 데이터 전처리 완료!")
        print("=" * 60)
        print(f"📊 결과:")
        print(f"  - Train: {len(X_train)}개")
        print(f"  - Val: {len(X_val)}개")
        print(f"  - 특징 차원: {X.shape[1]}차원")
        print(f"  - 출력 파일:")
        print(f"    * {output_dir / 'ml_training_dataset.npz'}")
        print(f"    * {output_dir / 'ml_dataset_metadata.json'}")
        print(f"    * {output_dir / 'training_sample.csv'}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
