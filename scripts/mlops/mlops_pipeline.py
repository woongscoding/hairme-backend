"""
MLOps 자동화 파이프라인

전체 프로세스:
1. DB에서 실제 사용자 데이터 추출
2. 합성 데이터와 병합
3. 모델 재학습
4. 성능 평가 및 배포

조건:
- 최소 피드백 데이터 개수 충족 시에만 실행
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pymysql


# 프로젝트 루트 디렉토리
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MLOpsPipeline:
    """MLOps 자동화 파이프라인"""

    def __init__(self, config=None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.project_root = project_root
        self.config = config or self._default_config()

        self.log_dir = self.project_root / "logs" / "mlops"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_log = []

    def _default_config(self):
        """기본 설정"""
        return {
            'min_feedback_count': 50,  # 최소 피드백 개수
            'real_data_weight': 2.0,   # 실제 데이터 가중치
            'min_improvement': 0.0,    # 최소 성능 개선폭
            'auto_deploy': True,       # 자동 배포 여부
            'batch_size': 64,
            'max_epochs': 50,
            'learning_rate': 0.001,
            'patience': 7
        }

    def log(self, message, level="INFO"):
        """로그 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.pipeline_log.append(log_entry)

    def check_feedback_count(self):
        """DB에 충분한 피드백 데이터가 있는지 확인"""
        self.log("피드백 데이터 개수 확인 중...")

        try:
            # 환경변수에서 DB 정보 가져오기
            database_url = os.getenv("DATABASE_URL")
            db_password = os.getenv("DB_PASSWORD")

            if not database_url or not db_password:
                self.log("환경변수 DATABASE_URL 또는 DB_PASSWORD가 없습니다.", "WARNING")
                return False

            # URL 파싱
            url = database_url.replace("asyncmy://", "")
            parts = url.split("@")
            user = parts[0]
            rest = parts[1]
            host_port, dbname = rest.split("/")

            if ":" in host_port:
                host, port_str = host_port.split(":")
                port = int(port_str)
            else:
                host = host_port
                port = 3306

            # DB 연결
            conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=db_password,
                database=dbname,
                charset='utf8mb4'
            )

            with conn.cursor() as cursor:
                query = """
                SELECT COUNT(*) as feedback_count
                FROM analysis_history
                WHERE style_1_feedback IS NOT NULL
                   OR style_2_feedback IS NOT NULL
                   OR style_3_feedback IS NOT NULL
                """
                cursor.execute(query)
                result = cursor.fetchone()
                feedback_count = result[0]

            conn.close()

            self.log(f"피드백 데이터: {feedback_count}건")
            self.log(f"최소 요구: {self.config['min_feedback_count']}건")

            if feedback_count >= self.config['min_feedback_count']:
                self.log(f"✅ 충분한 데이터 ({feedback_count}건 >= {self.config['min_feedback_count']}건)")
                return True
            else:
                self.log(f"⚠️ 데이터 부족 ({feedback_count}건 < {self.config['min_feedback_count']}건)", "WARNING")
                return False

        except Exception as e:
            self.log(f"피드백 확인 실패: {e}", "ERROR")
            return False

    def run_script(self, script_name, args=None):
        """Python 스크립트 실행"""
        script_path = self.project_root / "scripts" / "mlops" / script_name

        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        self.log(f"실행 중: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=3600  # 1시간 타임아웃
            )

            if result.returncode == 0:
                self.log(f"✅ {script_name} 성공")
                return True
            else:
                self.log(f"❌ {script_name} 실패 (exit code: {result.returncode})", "ERROR")
                self.log(f"STDERR: {result.stderr}", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"❌ {script_name} 타임아웃", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ {script_name} 실행 오류: {e}", "ERROR")
            return False

    def step_1_export_data(self):
        """1단계: 실제 데이터 추출"""
        self.log("=" * 60)
        self.log("Step 1: 실제 사용자 데이터 추출")
        self.log("=" * 60)

        return self.run_script("export_real_data.py")

    def step_2_prepare_data(self):
        """2단계: 학습 데이터 준비"""
        self.log("=" * 60)
        self.log("Step 2: 학습 데이터 준비")
        self.log("=" * 60)

        args = [
            "--real-weight", str(self.config['real_data_weight'])
        ]

        return self.run_script("prepare_training_data.py", args)

    def step_3_retrain_model(self):
        """3단계: 모델 재학습"""
        self.log("=" * 60)
        self.log("Step 3: 모델 재학습")
        self.log("=" * 60)

        args = [
            "--batch-size", str(self.config['batch_size']),
            "--epochs", str(self.config['max_epochs']),
            "--lr", str(self.config['learning_rate']),
            "--patience", str(self.config['patience'])
        ]

        return self.run_script("retrain_model.py", args)

    def step_4_deploy_model(self):
        """4단계: 모델 배포"""
        self.log("=" * 60)
        self.log("Step 4: 모델 평가 및 배포")
        self.log("=" * 60)

        args = [
            "--min-improvement", str(self.config['min_improvement'])
        ]

        if not self.config['auto_deploy']:
            args.append("--no-auto-deploy")

        return self.run_script("deploy_model.py", args)

    def save_pipeline_log(self):
        """파이프라인 로그 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{timestamp}.log"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.pipeline_log))

        self.log(f"로그 저장: {log_file}")

    def run(self, skip_data_check=False):
        """전체 파이프라인 실행"""
        self.log("🚀" * 30)
        self.log("MLOps 파이프라인 시작")
        self.log("🚀" * 30)

        start_time = datetime.now()

        try:
            # 0. 데이터 개수 확인 (옵션)
            if not skip_data_check:
                if not self.check_feedback_count():
                    self.log("파이프라인 중단: 데이터 부족", "WARNING")
                    self.save_pipeline_log()
                    return False
            else:
                self.log("데이터 개수 확인 스킵")

            # 1. 데이터 추출
            if not self.step_1_export_data():
                self.log("파이프라인 중단: 데이터 추출 실패", "ERROR")
                self.save_pipeline_log()
                return False

            # 2. 데이터 준비
            if not self.step_2_prepare_data():
                self.log("파이프라인 중단: 데이터 준비 실패", "ERROR")
                self.save_pipeline_log()
                return False

            # 3. 모델 재학습
            if not self.step_3_retrain_model():
                self.log("파이프라인 중단: 재학습 실패", "ERROR")
                self.save_pipeline_log()
                return False

            # 4. 모델 배포
            if not self.step_4_deploy_model():
                self.log("파이프라인 중단: 배포 실패", "ERROR")
                self.save_pipeline_log()
                return False

            # 성공
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.log("=" * 60)
            self.log(f"✅ MLOps 파이프라인 완료! (소요 시간: {duration:.1f}초)")
            self.log("=" * 60)

            self.save_pipeline_log()
            return True

        except Exception as e:
            self.log(f"파이프라인 오류: {e}", "ERROR")
            self.save_pipeline_log()
            return False


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="MLOps 자동화 파이프라인")
    parser.add_argument(
        "--min-feedback",
        type=int,
        default=50,
        help="최소 피드백 개수 (기본: 50)"
    )
    parser.add_argument(
        "--real-weight",
        type=float,
        default=2.0,
        help="실제 데이터 가중치 (기본: 2.0)"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="최소 성능 개선폭 (기본: 0.0)"
    )
    parser.add_argument(
        "--no-auto-deploy",
        action="store_true",
        help="자동 배포 비활성화"
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="데이터 개수 확인 스킵"
    )

    args = parser.parse_args()

    config = {
        'min_feedback_count': args.min_feedback,
        'real_data_weight': args.real_weight,
        'min_improvement': args.min_improvement,
        'auto_deploy': not args.no_auto_deploy,
        'batch_size': 64,
        'max_epochs': 50,
        'learning_rate': 0.001,
        'patience': 7
    }

    pipeline = MLOpsPipeline(config)
    success = pipeline.run(skip_data_check=args.skip_data_check)

    if success:
        print("\n🎉 파이프라인 성공!")
        return 0
    else:
        print("\n❌ 파이프라인 실패")
        return 1


if __name__ == "__main__":
    exit(main())
