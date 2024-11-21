import os
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# 환경 변수에서 Azure OpenAI 설정을 가져옵니다.
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"  # 실제 Azure API 버전

if not api_key or not azure_endpoint:
    raise ValueError("환경 변수 'AZURE_OPENAI_API_KEY'와 'AZURE_OPENAI_ENDPOINT'를 설정해야 합니다.")

# Azure OpenAI 클라이언트 초기화
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version,
)


class PromptRequest(BaseModel):
    prompt: str


class DeviceCategory(Enum):
    WASHING_MACHINE = "세탁기"
    DRYER = "건조기"
    REFRIGERATOR = "냉장고"
    AIR_CONDITIONER = "에어컨"
    TV = "TV"
    HUMIDIFIER = "가습기"
    AIR_PURIFIER = "공기청정기"
    OVEN = "오븐"
    KIMCHI_REFRIGERATOR = "김치냉장고"
    VACUUM_CLEANER = "청소기"


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


def parse_ai_response(ai_message):
    # DeviceCategory의 모든 값들을 가져옵니다.
    allowed_appliances = [device.value for device in DeviceCategory]

    # 응답을 줄 단위로 분리합니다.
    lines = ai_message.strip().split('\n')
    dialogues = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # 빈 줄은 스킵합니다.
        if ':' in line:
            # 첫 번째 콜론을 기준으로 가전제품명과 메시지를 분리합니다.
            appliance, message = line.split(':', 1)
            appliance = appliance.strip()
            message = message.strip()
            # 가전제품명이 허용된 목록에 있는지 확인합니다.
            if appliance in allowed_appliances:
                dialogues.append({'appliance': appliance, 'message': message})
            else:
                # 허용되지 않은 가전제품명인 경우, 에러 로그를 남기고 해당 대화를 스킵하거나 처리합니다.
                print(f"허용되지 않은 가전제품명 발견: {appliance}")
                continue
        else:
            # 포맷에 맞지 않는 경우 전체를 메시지로 저장합니다.
            dialogues.append({'appliance': None, 'message': line})
    return dialogues


@app.post("/generate")
async def generate_text(request: PromptRequest):
    try:
        # 현재 파일의 디렉토리를 기준으로 system_prompt.txt 파일의 경로를 설정합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(current_dir, 'system_prompt.txt')

        # system_prompt.txt 파일이 존재하는지 확인합니다.
        if not os.path.exists(system_prompt_path):
            raise FileNotFoundError(f"'{system_prompt_path}' 파일을 찾을 수 없습니다.")

        # system_prompt.txt 파일을 읽어옵니다.
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 배포된 모델의 이름
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # 파일에서 읽어온 프롬프트 내용 사용
                },
                {
                    "role": "user",
                    "content": request.prompt
                },
            ],
            max_tokens=50,
            temperature=0.5
        )

        # AI의 응답 메시지 내용만 추출
        ai_message = response.choices[0].message.content

        # 사용자 메시지를 대화 목록에 추가
        dialogues = [{'appliance': '사용자', 'message': request.prompt}]

        # AI의 응답을 파싱하여 대화 목록에 추가
        ai_dialogues = parse_ai_response(ai_message)
        dialogues.extend(ai_dialogues)

        return {"dialogues": dialogues}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
