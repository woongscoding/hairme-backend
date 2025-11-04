# init_db.py
import asyncio
from database import AsyncSessionLocal, engine, Base
from models import FaceShape, SkinTone, Hairstyle, Recommendation
from sqlalchemy import select

# 기존 HAIRSTYLE_RULES 데이터를 여기에 복사해옵니다.
HAIRSTYLE_RULES = {
    "계란형": {
        "쿨톤": ["시스루뱅 단발", "볼륨 웨이브 미디엄", "앞머리 없는 긴 생머리"],
        "웜톤": ["허쉬컷", "레이어드 롱", "C컬 웨이브 펌"],
        "중성톤": ["글램펌", "소프트 히피펌", "레이어드 C컬펌"],
    },
    "둥근형": {
        "쿨톤": ["사이드뱅 롱헤어", "앞머리 있는 단발", "볼륨매직"],
        "웜톤": ["S컬 웨이브", "허쉬컷", "긴 생머리"],
        "중성톤": ["단발 C컬펌", "레이어드 S컬펌", "사이드뱅"],
    },
    "각진형": {
        "쿨톤": ["굵은 웨이브 롱헤어", "사이드뱅 미디엄", "엘리자벳펌"],
        "웜톤": ["빌드펌", "긴머리 S컬펌", "사이드뱅"],
        "중성톤": ["보브컷", "미디엄 레이어드컷", "굵은 웨이브펌"],
    },
}


async def get_or_create(session, model, **kwargs):
    """
    DB에 데이터가 있으면 가져오고, 없으면 새로 생성합니다. (중복 방지)
    """
    result = await session.execute(select(model).filter_by(**kwargs))
    instance = result.scalars().first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        await session.flush()  # ID를 미리 받기 위해 flush
        return instance


async def main():
    print("🔄 데이터베이스 초기화 및 데이터 입력을 시작합니다...")

    # 테이블 생성 (혹시 모르니 한번 더 확인)
    async with engine.begin() as conn:
        # ❗️(주의) 아래 주석을 풀면 기존 데이터가 모두 삭제됩니다.
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        try:
            # 딕셔너리를 순회하며 DB에 삽입
            for shape_name, tones in HAIRSTYLE_RULES.items():
                # 1. 얼굴형 (FaceShape) 데이터
                face_shape = await get_or_create(session, FaceShape, name=shape_name)

                for tone_name, styles in tones.items():
                    # 2. 피부톤 (SkinTone) 데이터
                    skin_tone = await get_or_create(session, SkinTone, name=tone_name)

                    for i, style_name in enumerate(styles):
                        # 3. 헤어스타일 (Hairstyle) 데이터
                        hairstyle = await get_or_create(session, Hairstyle, name=style_name)

                        # 4. 추천 규칙 (Recommendation) 데이터
                        score = round(0.9 - i * 0.05, 2)
                        reason = f"{shape_name} 얼굴과 {tone_name} 피부에 잘 어울리는 스타일입니다."

                        # 중복 체크
                        stmt = select(Recommendation).where(
                            Recommendation.face_shape_id == face_shape.id,
                            Recommendation.skin_tone_id == skin_tone.id,
                            Recommendation.hairstyle_id == hairstyle.id
                        )
                        result = await session.execute(stmt)
                        existing_rec = result.scalars().first()

                        if not existing_rec:
                            recommendation = Recommendation(
                                face_shape_id=face_shape.id,
                                skin_tone_id=skin_tone.id,
                                hairstyle_id=hairstyle.id,
                                score=score,
                                reason=reason
                            )
                            session.add(recommendation)
                            print(f"  -> 추가: {shape_name} / {tone_name} / {style_name}")

            await session.commit()
            print("\n✅ 데이터베이스에 모든 추천 규칙을 성공적으로 입력했습니다.")

        except Exception as e:
            await session.rollback()
            print(f"\n❌ 데이터 입력 중 오류 발생: {e}")
        finally:
            await session.close()
            await engine.dispose()


if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때만 main() 함수가 호출됩니다.
    asyncio.run(main())