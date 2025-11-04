"""
합성 데이터와 실제 사용자 데이터를 병합하여 학습 데이터셋 생성

- 합성 데이터와 실제 데이터를 합침
- 학습/검증/테스트 세트로 분할
- 클래스 불균형 분석 및 리포트 생성
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split


# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TrainingDataPreparer:
    """학습 데이터 준비 클래스"""

    def __init__(
        self,
        synthetic_data_path="data_source/synthetic_hairstyle_data.csv",
        real_data_path="data_source/real_user_data_latest.csv",
        output_dir="data_source"
    ):
        """
        Args:
            synthetic_data_path: 합성 데이터 경로
            real_data_path: 실제 데이터 경로
            output_dir: 출력 디렉토리
        """
        self.project_root = project_root
        self.synthetic_path = self.project_root / synthetic_data_path
        self.real_path = self.project_root / real_data_path
        self.output_dir = self.project_root / output_dir

        self.synthetic_data = None
        self.real_data = None
        self.combined_data = None

    def load_data(self):
        """데이터 로드"""
        print("=" * 60)
        print("📂 데이터 로드 중...")
        print("=" * 60)

        # 합성 데이터 로드
        if self.synthetic_path.exists():
            self.synthetic_data = pd.read_csv(self.synthetic_path)
            print(f"✅ 합성 데이터 로드: {len(self.synthetic_data)}건")
        else:
            print(f"⚠️ 합성 데이터 파일 없음: {self.synthetic_path}")
            self.synthetic_data = pd.DataFrame()

        # 실제 데이터 로드
        if self.real_path.exists():
            self.real_data = pd.read_csv(self.real_path)
            print(f"✅ 실제 데이터 로드: {len(self.real_data)}건")
        else:
            print(f"⚠️ 실제 데이터 파일 없음: {self.real_path}")
            self.real_data = pd.DataFrame()

        if self.synthetic_data.empty and self.real_data.empty:
            raise ValueError("합성 데이터와 실제 데이터가 모두 비어있습니다!")

    def combine_data(self, real_data_weight=2.0):
        """
        데이터 병합

        Args:
            real_data_weight: 실제 데이터의 가중치 (실제 데이터를 몇 번 복제할지)
                             예: 2.0이면 실제 데이터를 2배로 증폭
        """
        print("\n" + "=" * 60)
        print("🔗 데이터 병합 중...")
        print("=" * 60)

        datasets = []

        # 합성 데이터 추가
        if not self.synthetic_data.empty:
            synthetic = self.synthetic_data.copy()
            synthetic['data_source'] = 'synthetic'
            datasets.append(synthetic)
            print(f"   합성 데이터: {len(synthetic)}건")

        # 실제 데이터 추가 (가중치 적용)
        if not self.real_data.empty:
            real = self.real_data.copy()
            real['data_source'] = 'real'

            # 실제 데이터 복제 (중요도 증폭)
            if real_data_weight > 1.0:
                replications = int(real_data_weight)
                for i in range(replications):
                    datasets.append(real.copy())
                print(f"   실제 데이터: {len(real)}건 × {replications} = {len(real) * replications}건")
            else:
                datasets.append(real)
                print(f"   실제 데이터: {len(real)}건")

        # 병합
        self.combined_data = pd.concat(datasets, ignore_index=True)

        # 셔플
        self.combined_data = self.combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✅ 병합 완료: 총 {len(self.combined_data)}건")

        # 통계 출력
        self._print_combined_statistics()

    def _print_combined_statistics(self):
        """병합된 데이터 통계 출력"""
        df = self.combined_data

        print("\n📊 병합 데이터 통계:")
        print(f"   총 레코드: {len(df)}건")

        if 'data_source' in df.columns:
            print(f"\n   데이터 출처:")
            print(df['data_source'].value_counts().to_string())

        print(f"\n   얼굴형 분포:")
        print(df['face_shape'].value_counts().to_string())

        print(f"\n   피부톤 분포:")
        print(df['skin_tone'].value_counts().to_string())

        print(f"\n   헤어스타일 분포:")
        print(df['hairstyle'].value_counts().to_string())

        print(f"\n   피드백 분포:")
        feedback_counts = df['feedback'].value_counts()
        print(feedback_counts.to_string())

        # 클래스 불균형 비율
        if 'like' in feedback_counts.index and 'dislike' in feedback_counts.index:
            like_count = feedback_counts['like']
            dislike_count = feedback_counts['dislike']
            imbalance_ratio = like_count / dislike_count if dislike_count > 0 else float('inf')
            print(f"\n   ⚠️ 클래스 불균형 비율 (like/dislike): {imbalance_ratio:.2f}")

            if imbalance_ratio > 3.0:
                print(f"   ⚠️ 경고: like가 dislike보다 {imbalance_ratio:.1f}배 많습니다.")
                print(f"         재학습 시 class_weight를 적용하는 것을 권장합니다.")

        print(f"\n   네이버 클릭률: {df['naver_clicked'].mean()*100:.1f}%")
        print(f"   평균 점수: {df['score'].mean():.3f}")

    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        학습/검증/테스트 세트 분할

        Args:
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
        """
        print("\n" + "=" * 60)
        print("✂️ 데이터 분할 중...")
        print("=" * 60)

        # 비율 검증
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"비율의 합이 1.0이 아닙니다: {total_ratio}")

        df = self.combined_data

        # 1차 분할: train + (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=df['feedback']  # 피드백 비율 유지
        )

        # 2차 분할: val + test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_df['feedback']
        )

        print(f"✅ 학습 데이터: {len(train_df)}건 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"✅ 검증 데이터: {len(val_df)}건 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"✅ 테스트 데이터: {len(test_df)}건 ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def save_datasets(self, train_df, val_df, test_df):
        """데이터셋 저장"""
        print("\n" + "=" * 60)
        print("💾 데이터셋 저장 중...")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 타임스탬프 버전
        train_path = self.output_dir / f"train_data_{timestamp}.csv"
        val_path = self.output_dir / f"val_data_{timestamp}.csv"
        test_path = self.output_dir / f"test_data_{timestamp}.csv"

        # 최신 버전 (심볼릭 링크처럼 사용)
        train_latest = self.output_dir / "train_data.csv"
        val_latest = self.output_dir / "val_data.csv"
        test_latest = self.output_dir / "test_data.csv"

        # data_source 컬럼 제거 (학습에 불필요)
        train_clean = train_df.drop(columns=['data_source'], errors='ignore')
        val_clean = val_df.drop(columns=['data_source'], errors='ignore')
        test_clean = test_df.drop(columns=['data_source'], errors='ignore')

        # 저장
        train_clean.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_clean.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_clean.to_csv(test_path, index=False, encoding='utf-8-sig')

        train_clean.to_csv(train_latest, index=False, encoding='utf-8-sig')
        val_clean.to_csv(val_latest, index=False, encoding='utf-8-sig')
        test_clean.to_csv(test_latest, index=False, encoding='utf-8-sig')

        print(f"✅ 저장 완료:")
        print(f"   {train_path}")
        print(f"   {val_path}")
        print(f"   {test_path}")

        # 리포트 생성
        self._generate_report(train_df, val_df, test_df, timestamp)

        return {
            'train': train_path,
            'val': val_path,
            'test': test_path
        }

    def _generate_report(self, train_df, val_df, test_df, timestamp):
        """데이터 준비 리포트 생성"""
        report_path = self.output_dir / f"data_preparation_report_{timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("데이터 준비 리포트\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 전체 통계
            f.write("## 전체 데이터 통계\n")
            f.write(f"총 레코드: {len(self.combined_data)}건\n\n")

            if 'data_source' in self.combined_data.columns:
                f.write("데이터 출처:\n")
                f.write(self.combined_data['data_source'].value_counts().to_string())
                f.write("\n\n")

            # 분할 통계
            f.write("## 데이터 분할\n")
            f.write(f"학습 데이터: {len(train_df)}건\n")
            f.write(f"검증 데이터: {len(val_df)}건\n")
            f.write(f"테스트 데이터: {len(test_df)}건\n\n")

            # 각 세트의 피드백 분포
            for name, df in [("학습", train_df), ("검증", val_df), ("테스트", test_df)]:
                f.write(f"### {name} 데이터 피드백 분포\n")
                f.write(df['feedback'].value_counts().to_string())
                f.write("\n\n")

        print(f"📄 리포트 생성: {report_path}")

    def prepare(self, real_data_weight=2.0):
        """전체 준비 프로세스 실행"""
        print("\n" + "🚀" * 30)
        print("학습 데이터 준비 시작")
        print("🚀" * 30 + "\n")

        # 1. 데이터 로드
        self.load_data()

        # 2. 데이터 병합
        self.combine_data(real_data_weight=real_data_weight)

        # 3. 데이터 분할
        train_df, val_df, test_df = self.split_data()

        # 4. 저장
        paths = self.save_datasets(train_df, val_df, test_df)

        print("\n" + "=" * 60)
        print("✅ 데이터 준비 완료!")
        print("=" * 60)

        return paths


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="학습 데이터 준비")
    parser.add_argument(
        "--real-weight",
        type=float,
        default=2.0,
        help="실제 데이터 가중치 (기본: 2.0, 실제 데이터를 2배로 증폭)"
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        default="data_source/synthetic_hairstyle_data.csv",
        help="합성 데이터 경로"
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="data_source/real_user_data_latest.csv",
        help="실제 데이터 경로"
    )

    args = parser.parse_args()

    try:
        preparer = TrainingDataPreparer(
            synthetic_data_path=args.synthetic_data,
            real_data_path=args.real_data
        )

        paths = preparer.prepare(real_data_weight=args.real_weight)

        print("\n✅ 생성된 파일:")
        for key, path in paths.items():
            print(f"   {key}: {path}")

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
