import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# 환경 변수에서 Azure OpenAI 설정을 가져옵니다.
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"  # 실제 Azure API 버전을 입력

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


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/generate")
async def generate_text(request: PromptRequest):
    print(api_key, azure_endpoint, api_version)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 배포된 모델의 이름으로 변경해야 합니다.
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 스마트 홈의 여러 가전제품 역할을 하는 AI 비서입니다. "
                        "사용자의 요청에 따라 적절한 가전제품으로서 응답하세요. "
                        "응답 시 다음 형식을 따르세요:\n\n"
                        "가전제품명: 응답 내용\n\n"
                        "예시:\n"
                        "사용자: 냉장고야, 우유가 있니?\n"
                        "냉장고: 현재 우유가 두 개 남아 있습니다.\n\n"
                        "사용자: 방이 좀 더운 거 같아\n"
                        "에어컨: 마지막으로 설정한 온도(21도)로 냉방을 시작할까요?\n\n"
                        "사용자: 세탁기야, 빨래 다 됐어?\n"
                        "세탁기: 네, 세탁이 완료되었습니다.\n\n"
                        "만약 사용자가 특정 가전제품에게 말했지만 다른 가전제품이 더 적절하다면, "
                        "그 가전제품으로서 응답하고 자연스럽게 전환을 알려주세요."
                    )
                },

                {
                    "role": "user",
                    "content": request.prompt
                },
            ],
            max_tokens=150,
            temperature=0.5
        )

        # AI의 응답 메시지 내용만 추출
        ai_message = response.choices[0].message.content

        return {"response": ai_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
