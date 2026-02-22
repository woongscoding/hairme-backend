#!/usr/bin/env python3
"""
데이터 수집 진행 상황 모니터링

실시간으로 로그 파일을 읽어서 진행 상황을 표시합니다.

사용법:
  python scripts/monitor_progress.py logs/collection_*.log
"""

import sys
import time
import re
from pathlib import Path
import glob


def monitor_log(log_file):
    """로그 파일 실시간 모니터링"""
    print(f"📊 모니터링 중: {log_file}")
    print("=" * 60)
    print()

    # 통계 초기화
    stats = {"total": 0, "success": 0, "failed": 0, "samples": 0}

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            # 파일 끝으로 이동
            f.seek(0, 2)

            while True:
                line = f.readline()

                if not line:
                    time.sleep(0.5)
                    continue

                # 진행률 파싱
                if "진행률:" in line or "AI 얼굴 처리 중" in line:
                    print(line.strip())

                # 성공/실패 카운트
                if "성공:" in line and "샘플 생성" in line:
                    match = re.search(r"(\d+)개 샘플", line)
                    if match:
                        stats["samples"] += int(match.group(1))
                        stats["success"] += 1

                if "MediaPipe 실패" in line or "Gemini 실패" in line:
                    stats["failed"] += 1

                # 통계 업데이트
                if "📊 진행률:" in line:
                    match = re.search(r"\[(\d+)/(\d+)\]", line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        stats["total"] = total

                        # 5의 배수마다 통계 출력
                        if current % 5 == 0:
                            print()
                            print(f"📊 현재 통계:")
                            print(
                                f"  진행: {current}/{total} ({current/total*100:.1f}%)"
                            )
                            print(f"  성공: {stats['success']}개")
                            print(f"  실패: {stats['failed']}개")
                            print(f"  누적 샘플: {stats['samples']}개")
                            print()

                # 완료 메시지
                if "수집 완료" in line or "✅" in line:
                    print()
                    print("=" * 60)
                    print(line.strip())
                    print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n모니터링 종료")
        return
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return


def find_latest_log():
    """가장 최근 로그 파일 찾기"""
    log_files = glob.glob("logs/collection_*.log")

    if not log_files:
        return None

    # 수정 시간 기준 정렬
    log_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return log_files[0]


def main():
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # 자동으로 최신 로그 찾기
        log_file = find_latest_log()

        if not log_file:
            print("❌ 로그 파일을 찾을 수 없습니다.")
            print("\n사용법:")
            print("  python scripts/monitor_progress.py logs/collection_*.log")
            return

        print(f"✅ 최신 로그 파일 찾음: {log_file}")
        print()

    if not Path(log_file).exists():
        print(f"❌ 파일이 없습니다: {log_file}")
        return

    monitor_log(log_file)


if __name__ == "__main__":
    main()
