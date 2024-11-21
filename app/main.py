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


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/generate")
async def generate_text(request: PromptRequest):
    print(api_key, azure_endpoint, api_version)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 배포된 모델의 이름
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 스마트 홈의 여러 가전제품 역할을 하는 AI 비서입니다. "
                        "사용자가 특정 가전제품을 호출하면, 그 가전제품으로서 먼저 응답하세요. "
                        "만약 요청이 다른 가전제품이 더 적절한 경우라면, 호출된 가전제품이 먼저 그 사실을 알려주고, "
                        "필요한 경우 해당 가전제품으로 전환하여 응답합니다. "
                        "응답 시 다음 형식을 따르세요:\n\n"
                        "가전제품명: 응답 내용\n\n"
                        "예시:\n"
                        "사용자: 냉장고야, 방이 좀 더운 거 같아\n"
                        "냉장고: 제가 그 부분은 잘 못할 거 같네요. 에어컨에게 요청해 볼게요.\n"
                        "에어컨: 마지막으로 설정한 온도(21도)로 냉방을 시작할까요?\n\n"
                        "사용자: 세탁기야, 우유가 있니?\n"
                        "세탁기: 그 부분은 냉장고가 더 잘 알 것 같습니다.\n"
                        "냉장고: 현재 우유가 두 개 남아 있습니다.\n\n"
                        "사용자: 세탁기야, 공기가 좀 안 좋은 것 같아\n"
                        "세탁기: 제가 그 부분은 잘 못하지만, 공기청정기가 도와드릴 수 있어요.\n"
                        "공기청정기: 현재 공기 질이 좋지 않습니다. 작동을 시작할까요?\n\n"
                        "사용자: 에어컨아, 온도를 23도로 올려줘\n"
                        "에어컨: 온도를 23도로 설정했습니다.\n\n"
                        "사용자: 거실이 너무 습한 것 같아."
                        "공기청정기: 거실의 공기 질도 나쁘네요. 공기청정 모드와 제습 모드를 동시에 작동할까요?"
                        "항상 호출된 가전제품이 먼저 응답하고, 필요한 경우 다른 가전제품으로 자연스럽게 전환하세요."
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
