#!/usr/bin/env python3
"""
AWS 비용 검증 스크립트
Cost Explorer API로 실제 비용 절감 확인

실행 방법:
    python scripts/cost_verification.py

환경 변수:
    AWS_REGION (선택사항, 기본값: ap-northeast-2)

참고:
    - Cost Explorer API는 us-east-1 리전만 지원
    - AWS 계정에 Cost Explorer가 활성화되어 있어야 함
    - IAM 권한 필요: ce:GetCostAndUsage
"""

import os
import sys
import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from decimal import Decimal

# 컬러 출력
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    """헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def print_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


class CostAnalyzer:
    """AWS 비용 분석기"""

    def __init__(self):
        """초기화"""
        self.region = os.getenv('AWS_REGION', 'ap-northeast-2')

        # Cost Explorer는 us-east-1만 지원
        try:
            self.ce = boto3.client('ce', region_name='us-east-1')
            self.sts = boto3.client('sts', region_name=self.region)

            # 계정 정보 확인
            identity = self.sts.get_caller_identity()
            self.account_id = identity['Account']
            print_info(f"AWS 계정: {self.account_id}")
            print_info(f"분석 리전: {self.region}\n")

        except Exception as e:
            print_error(f"AWS 클라이언트 초기화 실패: {str(e)}")
            sys.exit(1)

    def get_current_month_cost(self) -> Dict[str, float]:
        """현재 월 비용 조회"""
        print_header("📊 현재 월 비용 조회")

        today = datetime.now()
        first_of_month = today.replace(day=1)

        try:
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': first_of_month.strftime('%Y-%m-%d'),
                    'End': today.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )

            costs = {}
            total = 0.0

            if response['ResultsByTime']:
                for group in response['ResultsByTime'][0]['Groups']:
                    service = group['Keys'][0]
                    amount = float(group['Metrics']['UnblendedCost']['Amount'])

                    if amount > 0.01:  # 1센트 이상만
                        costs[service] = amount
                        total += amount

            print_success(f"현재 월 총 비용: ${total:.2f}")
            print_info(f"기간: {first_of_month.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}")
            print_info(f"서비스 수: {len(costs)}개\n")

            return {
                'total': total,
                'by_service': costs,
                'period_start': first_of_month.strftime('%Y-%m-%d'),
                'period_end': today.strftime('%Y-%m-%d')
            }

        except Exception as e:
            print_error(f"현재 월 비용 조회 실패: {str(e)}")
            return {'total': 0.0, 'by_service': {}}

    def get_last_month_cost(self) -> Dict[str, float]:
        """전월 비용 조회"""
        print_header("📊 전월 비용 조회")

        today = datetime.now()
        first_of_month = today.replace(day=1)
        last_month_end = first_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        try:
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': last_month_start.strftime('%Y-%m-%d'),
                    'End': first_of_month.strftime('%Y-%m-%d')  # 전월 마지막 날 다음날
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )

            costs = {}
            total = 0.0

            if response['ResultsByTime']:
                for group in response['ResultsByTime'][0]['Groups']:
                    service = group['Keys'][0]
                    amount = float(group['Metrics']['UnblendedCost']['Amount'])

                    if amount > 0.01:
                        costs[service] = amount
                        total += amount

            print_success(f"전월 총 비용: ${total:.2f}")
            print_info(f"기간: {last_month_start.strftime('%Y-%m-%d')} ~ {last_month_end.strftime('%Y-%m-%d')}")
            print_info(f"서비스 수: {len(costs)}개\n")

            return {
                'total': total,
                'by_service': costs,
                'period_start': last_month_start.strftime('%Y-%m-%d'),
                'period_end': last_month_end.strftime('%Y-%m-%d')
            }

        except Exception as e:
            print_error(f"전월 비용 조회 실패: {str(e)}")
            return {'total': 0.0, 'by_service': {}}

    def get_service_breakdown(self, days: int = 7) -> Dict[str, float]:
        """최근 N일간 서비스별 비용 분석"""
        print_header(f"📈 최근 {days}일 서비스별 비용 분석")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )

            # 서비스별 합계 계산
            service_totals = {}

            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    amount = float(group['Metrics']['UnblendedCost']['Amount'])

                    if service not in service_totals:
                        service_totals[service] = 0.0
                    service_totals[service] += amount

            # 비용 순으로 정렬
            sorted_services = sorted(
                service_totals.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # 상위 10개 서비스 출력
            print(f"\n{'서비스':<40} {'비용':>15}")
            print("-" * 56)

            for service, cost in sorted_services[:10]:
                if cost > 0.01:
                    print(f"{service:<40} ${cost:>14.2f}")

            total = sum(service_totals.values())
            print("-" * 56)
            print(f"{'총계':<40} ${total:>14.2f}\n")

            return {
                'by_service': dict(sorted_services),
                'total': total,
                'period_days': days
            }

        except Exception as e:
            print_error(f"서비스별 비용 조회 실패: {str(e)}")
            return {'by_service': {}, 'total': 0.0}

    def calculate_savings(self, last_month: Dict, current_month: Dict) -> Dict:
        """절감액 계산 및 비교"""
        print_header("💰 비용 절감 분석")

        last_total = last_month.get('total', 0.0)
        current_total = current_month.get('total', 0.0)

        # 현재 월은 진행 중이므로 일할 계산
        today = datetime.now()
        days_in_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        days_in_month = days_in_month.day
        current_day = today.day

        # 월 전체 예상 비용
        estimated_current_month = (current_total / current_day) * days_in_month if current_day > 0 else current_total

        savings = last_total - estimated_current_month
        savings_percent = (savings / last_total * 100) if last_total > 0 else 0

        print(f"{Colors.BOLD}전월 비용 (완료):{Colors.RESET}          ${last_total:.2f}")
        print(f"{Colors.BOLD}현재 월 비용 (진행 중):{Colors.RESET}    ${current_total:.2f} ({current_day}/{days_in_month}일)")
        print(f"{Colors.BOLD}현재 월 예상 비용:{Colors.RESET}         ${estimated_current_month:.2f}\n")

        if savings > 0:
            print_success(f"예상 절감액: ${savings:.2f} ({savings_percent:.1f}% 감소)")
            print_success(f"연간 예상 절감: ${savings * 12:.2f}\n")
        elif savings < 0:
            print_warning(f"예상 증가액: ${abs(savings):.2f} ({abs(savings_percent):.1f}% 증가)")
            print_warning(f"연간 예상 증가: ${abs(savings) * 12:.2f}\n")
        else:
            print_info("비용 변화 없음\n")

        # 서비스별 비교
        print(f"\n{Colors.BOLD}서비스별 비용 변화:{Colors.RESET}\n")
        print(f"{'서비스':<30} {'전월':>12} {'현재':>12} {'변화':>12}")
        print("-" * 68)

        all_services = set(
            list(last_month.get('by_service', {}).keys()) +
            list(current_month.get('by_service', {}).keys())
        )

        # 주요 서비스만 표시
        important_services = [
            'Amazon Relational Database Service',
            'Amazon DynamoDB',
            'AWS Lambda',
            'Amazon Elastic Compute Cloud - Compute',
            'Amazon Elastic Container Service',
            'Amazon EC2 Container Registry (ECR)',
            'Amazon Virtual Private Cloud',
            'Amazon Simple Storage Service',
            'Amazon API Gateway',
            'EC2 - Other'
        ]

        for service in important_services:
            if service in all_services:
                last_cost = last_month.get('by_service', {}).get(service, 0.0)
                curr_cost = current_month.get('by_service', {}).get(service, 0.0)

                # 일할 계산
                estimated_curr = (curr_cost / current_day) * days_in_month if current_day > 0 else curr_cost
                change = estimated_curr - last_cost

                change_str = f"{change:+.2f}"
                if change > 0:
                    change_color = Colors.RED
                elif change < 0:
                    change_color = Colors.GREEN
                else:
                    change_color = Colors.RESET

                # 서비스명 단축
                short_name = service.replace('Amazon ', '').replace('AWS ', '')
                if len(short_name) > 28:
                    short_name = short_name[:25] + "..."

                print(f"{short_name:<30} ${last_cost:>10.2f} ${estimated_curr:>10.2f} {change_color}${change_str:>10}{Colors.RESET}")

        print("-" * 68)
        print(f"{'총계':<30} ${last_total:>10.2f} ${estimated_current_month:>10.2f} ${savings:>10.2f}\n")

        return {
            'last_month_total': last_total,
            'current_month_actual': current_total,
            'current_month_estimated': estimated_current_month,
            'savings': savings,
            'savings_percent': savings_percent,
            'annual_savings': savings * 12
        }

    def generate_cost_report(self, last_month: Dict, current_month: Dict,
                           breakdown: Dict, savings: Dict) -> str:
        """비용 리포트 마크다운 생성"""
        print_header("📄 비용 리포트 생성")

        report = f"""# 💰 HairMe 비용 분석 리포트

## 생성 정보
- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **AWS 계정**: {self.account_id}
- **분석 리전**: {self.region}

---

## 📊 월별 비용 비교

### 전월 ({last_month.get('period_start', 'N/A')} ~ {last_month.get('period_end', 'N/A')})
- **총 비용**: ${last_month.get('total', 0):.2f}
- **서비스 수**: {len(last_month.get('by_service', {}))}개

### 현재 월 ({current_month.get('period_start', 'N/A')} ~ {current_month.get('period_end', 'N/A')})
- **현재까지 비용**: ${current_month.get('total', 0):.2f}
- **예상 월말 비용**: ${savings.get('current_month_estimated', 0):.2f}
- **서비스 수**: {len(current_month.get('by_service', {}))}개

---

## 💸 비용 절감 분석

| 항목 | 금액 |
|------|------|
| 전월 총 비용 | ${savings.get('last_month_total', 0):.2f} |
| 현재 월 예상 비용 | ${savings.get('current_month_estimated', 0):.2f} |
| **절감액** | **${savings.get('savings', 0):.2f}** |
| **절감률** | **{savings.get('savings_percent', 0):.1f}%** |
| **연간 예상 절감** | **${savings.get('annual_savings', 0):.2f}** |

"""

        # 절감 상태
        if savings.get('savings', 0) > 0:
            report += f"""
### ✅ 비용 절감 성공!

마이그레이션으로 인해 **월 ${savings.get('savings', 0):.2f}**의 비용 절감이 예상됩니다.
이는 연간 약 **${savings.get('annual_savings', 0):.2f}**의 절감 효과입니다.
"""
        elif savings.get('savings', 0) < 0:
            report += f"""
### ⚠️ 비용 증가 감지

현재 월 예상 비용이 전월 대비 **${abs(savings.get('savings', 0)):.2f}** 증가했습니다.
원인을 분석하고 최적화가 필요합니다.
"""
        else:
            report += "\n### ℹ️ 비용 변화 없음\n\n"

        # 서비스별 비용 상세
        report += f"""
---

## 📈 서비스별 비용 상세 (최근 {breakdown.get('period_days', 7)}일)

| 서비스 | 비용 | 비율 |
|--------|------|------|
"""

        total = breakdown.get('total', 0)
        for service, cost in list(breakdown.get('by_service', {}).items())[:15]:
            if cost > 0.01:
                percentage = (cost / total * 100) if total > 0 else 0
                report += f"| {service} | ${cost:.2f} | {percentage:.1f}% |\n"

        report += f"""
| **총계** | **${total:.2f}** | **100%** |

---

## 🎯 비용 최적화 권장 사항

"""

        # 서비스별 권장 사항
        recommendations = []

        # RDS 체크
        rds_cost = current_month.get('by_service', {}).get('Amazon Relational Database Service', 0)
        if rds_cost > 1.0:
            recommendations.append(
                f"- ⚠️ **RDS 비용**: ${rds_cost:.2f} - RDS가 아직 실행 중입니다. DynamoDB 마이그레이션 후 삭제를 고려하세요."
            )

        # DynamoDB 체크
        dynamodb_cost = current_month.get('by_service', {}).get('Amazon DynamoDB', 0)
        if dynamodb_cost > 0:
            recommendations.append(
                f"- ✅ **DynamoDB 비용**: ${dynamodb_cost:.2f} - DynamoDB가 정상 작동 중입니다."
            )

        # Lambda 체크
        lambda_cost = current_month.get('by_service', {}).get('AWS Lambda', 0)
        if lambda_cost > 5.0:
            recommendations.append(
                f"- ⚠️ **Lambda 비용**: ${lambda_cost:.2f} - Lambda 실행 시간 최적화를 고려하세요."
            )

        # NAT Gateway 체크
        vpc_cost = current_month.get('by_service', {}).get('Amazon Virtual Private Cloud', 0)
        if vpc_cost > 10.0:
            recommendations.append(
                f"- ⚠️ **VPC 비용**: ${vpc_cost:.2f} - NAT Gateway가 아직 실행 중일 수 있습니다. 확인 후 삭제하세요."
            )

        # EC2 체크
        ec2_cost = current_month.get('by_service', {}).get('Amazon Elastic Compute Cloud - Compute', 0)
        if ec2_cost > 0:
            recommendations.append(
                f"- ℹ️ **EC2 비용**: ${ec2_cost:.2f} - ECS 또는 기타 서비스 사용 중"
            )

        if recommendations:
            for rec in recommendations:
                report += rec + "\n"
        else:
            report += "- ✅ 현재 비용 구조가 최적화되어 있습니다.\n"

        report += """
---

## 📋 다음 단계

1. [ ] RDS 인스턴스 삭제 (DynamoDB 마이그레이션 완료 시)
2. [ ] NAT Gateway 삭제 (불필요시)
3. [ ] ALB 삭제 (Lambda/API Gateway 사용 시)
4. [ ] CloudWatch 로그 보관 기간 조정
5. [ ] S3 라이프사이클 정책 설정
6. [ ] Lambda 메모리 최적화
7. [ ] DynamoDB Auto Scaling 설정

---

**보고서 끝**

*이 리포트는 AWS Cost Explorer API를 통해 자동 생성되었습니다.*
"""

        # 파일 저장
        report_path = 'COST_ANALYSIS_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print_success(f"비용 리포트 저장: {report_path}\n")

        return report

    def analyze(self):
        """전체 비용 분석 실행"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                                                                    ║")
        print("║              💰 HairMe AWS 비용 분석                                ║")
        print("║              Cost Explorer Analysis                               ║")
        print("║                                                                    ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}\n")

        try:
            # 1. 현재 월 비용
            current_month = self.get_current_month_cost()

            # 2. 전월 비용
            last_month = self.get_last_month_cost()

            # 3. 서비스별 분석 (최근 7일)
            breakdown = self.get_service_breakdown(days=7)

            # 4. 절감액 계산
            savings = self.calculate_savings(last_month, current_month)

            # 5. 리포트 생성
            report = self.generate_cost_report(last_month, current_month, breakdown, savings)

            # 6. JSON 출력 (자동화 용)
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'account_id': self.account_id,
                'last_month': last_month,
                'current_month': current_month,
                'savings': savings,
                'breakdown': breakdown
            }

            with open('cost_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print_success("JSON 데이터 저장: cost_analysis.json\n")

            # 최종 요약
            print_header("🎯 분석 완료")

            if savings.get('savings', 0) > 0:
                print_success(f"월간 절감: ${savings['savings']:.2f} ({savings['savings_percent']:.1f}%)")
                print_success(f"연간 절감: ${savings['annual_savings']:.2f}")
                print("\n✅ 마이그레이션이 비용 절감에 성공했습니다!\n")
            elif savings.get('savings', 0) < 0:
                print_warning(f"월간 증가: ${abs(savings['savings']):.2f}")
                print("\n⚠️ 비용이 증가했습니다. 최적화가 필요합니다.\n")
            else:
                print_info("비용 변화 없음\n")

        except Exception as e:
            print_error(f"분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """메인 함수"""
    analyzer = CostAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
