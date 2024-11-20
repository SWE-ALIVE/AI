from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI

load_dotenv()

app = FastAPI()

# 환경 변수에서 API 키와 엔드포인트를 가져옵니다.
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-02-15-preview"
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not azure_endpoint:
    raise ValueError(
        "환경 변수 'AZURE_OPENAI_API_KEY'와 'AZURE_OPENAI_ENDPOINT'를 설정해야 합니다."
    )

# AzureOpenAI 클라이언트 초기화
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
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
                    "role": "user",
                    "content": "What steps should I think about when writing my first Python API?",
                },
            ],
            max_tokens=50,
        )
        return {"response": response.model_dump_json(indent=2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
