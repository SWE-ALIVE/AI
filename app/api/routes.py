from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import os
from openai import OpenAI
from app.retriever import KoreanDialogRetriever
from app.helper.turn_data_to_str import turn_data_to_str
from app.config.data_class import APIRequest, APIResponse
import json

router = APIRouter()

domain_slot_file = "app/config/domain_slots.json"
with open(domain_slot_file, "r") as f:
    DOMAIN_SLOTS = json.load(f)


retriever = KoreanDialogRetriever()
retriever.load_dialogs("app/data/gpt4_dataset.json")
retriever.build_index()


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


table_prompt = load_prompt("app/prompts/table_prompt.txt")
print("✅ 테이블 프롬프트 로딩 완료")


@router.post("/generate_sql", response_model=APIResponse)
def generate_sql(request: APIRequest):
    examples = retriever.retrieve(request.model_dump(), top_k=3)
    prompt = (
        table_prompt
        + examples
        + "Example #4\n"
        + turn_data_to_str(request.model_dump())
        + "\nSQL: "
    )
    print(f"✅ SQL 생성 프롬프트 생성 완료: {prompt}")

    system_prompt = f"""
    당신은 사용자가 제어하려는 가전제품에 대해 자연스러운 대화를 진행하고, 대화 상태를 기반으로 SQL 쿼리를 생성하는 AI입니다. 아래 내용을 참고하여 각 가전제품의 동작을 처리하세요:
    직전 대화
    - 사용자: {request.dialog['usr'][-1]}
    - 시스템: {request.dialog['sys'][-1]}
    생성 규칙
    1. **context**:
    - 사용자의 발화 내용으로 인해 변경된 슬롯 값 목록입니다. 이때 도메인은 다음 중 하나이어야 합니다: {", ".join(DOMAIN_SLOTS.keys())}
    3. **message**:
    - 해당 의인화된 가전제품이 직접 사용자와 대화하는 것처럼 자연스럽게 응답합니다.
    4. **sql**:
    - 대화 상태를 반영한 SQL 쿼리를 생성합니다.
    - 예: "UPDATE AIR_PURIFIER SET fan_speed = "보통" WHERE name = '휘센 오브제컬렉션 위너 에어컨';"
    """

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
            response_format=APIResponse,
        )

        if completion:
            response = completion.choices[0].message.parsed
            return response
        else:
            raise HTTPException(status_code=500, detail="No completion received.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
