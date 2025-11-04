import pymysql
import os

# RDS 연결 정보
DB_HOST = "hairme-data.cr28a6uqo2k8.ap-northeast-2.rds.amazonaws.com"
DB_USER = "admin"
DB_PASSWORD = input("DB 비밀번호 입력: ")
DB_NAME = "hairme"

try:
    # 연결
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4'
    )
    
    cursor = conn.cursor()
    
    # 스키마 업데이트 실행
    with open('db_schema_v20.sql', 'r', encoding='utf-8') as f:
        sql_commands = f.read().split(';')
        
        for command in sql_commands:
            command = command.strip()
            if command and not command.startswith('--') and command != 'USE hairme':
                try:
                    print(f"실행 중: {command[:50]}...")
                    cursor.execute(command)
                    print("✅ 성공")
                except Exception as e:
                    print(f"⚠️ 경고: {str(e)}")
    
    conn.commit()
    print("\n✅ 스키마 업데이트 완료!")
    
    # 테이블 구조 확인
    cursor.execute("DESCRIBE analysis_history")
    columns = cursor.fetchall()
    
    print("\n현재 테이블 구조:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ 오류: {str(e)}")
