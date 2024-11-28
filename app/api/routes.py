from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import os
from openai import OpenAI
from app.retriever import KoreanDialogRetriever
from app.helper.turn_data_to_str import turn_data_to_str
import json

router = APIRouter()

domain_slot_file = "app/data/domain_slots.json"
with open(domain_slot_file, "r") as f:
    DOMAIN_SLOTS = json.load(f)


class TurnData(BaseModel):
    dialog: Dict[str, List[str]]
    slot_values: List[Dict[str, str]]


class APIResponse(BaseModel):
    sql: str
    message: str
    context: str
    domain: str


retriever = KoreanDialogRetriever()
retriever.load_dialogs("app/data/dataset.json")
retriever.build_index()


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


system_prompt = load_prompt("app/prompts/system_prompt.txt")
table_prompt = load_prompt("app/prompts/table_prompt.txt")


@router.post("/generate_sql", response_model=APIResponse)
def generate_sql(turn_data: TurnData):
    examples = retriever.retrieve(turn_data.model_dump(), top_k=3)
    prompt = (
        table_prompt
        + examples
        + "Example #6\n"
        + turn_data_to_str(turn_data.model_dump())
        + "\nSQL: "
    )

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
            turn_data = completion.choices[0].message.parsed
            return APIResponse(
                sql=turn_data.sql,
                message=turn_data.message,
                context=turn_data.context,
                domain=turn_data.domain,
            )
        else:
            raise HTTPException(status_code=500, detail="No completion received.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
