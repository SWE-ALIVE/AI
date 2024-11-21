import os
from openai import AzureOpenAI
from pydantic import BaseModel
from typing import Literal
import json
import time
import random

domain_slot_file = "data/domain_slots.json"
output_file = "data/dataset_azure.json"

with open(domain_slot_file, "r") as f:
    DOMAIN_SLOTS = json.load(f)

DOMAIN_TYPES = Literal[tuple(DOMAIN_SLOTS.keys())]
SLOT_TYPES = {domain: Literal[tuple(slots)] for domain, slots in DOMAIN_SLOTS.items()}


class Dialog(BaseModel):
    sys: list[str]
    usr: list[str]


class SlotValue(BaseModel):
    domain: str
    slot: str
    value: str


class TurnData(BaseModel):
    ID: str
    turn_id: int
    domains: list[str]
    dialog: Dialog
    slot_values: list[SlotValue]
    turn_slot_values: list[SlotValue]
    last_slot_values: list[SlotValue]


def create_system_prompt(
    domains: list[str], previous_turn: dict = None, is_last_turn: bool = False
) -> str:
    # 현재 도메인에 대한 슬롯 정보만 포함
    available_slots = {domain: DOMAIN_SLOTS[domain] for domain in domains}

    base_prompt = (
        "당신은 다중 도메인 작업 지향 대화를 생성하는 AI입니다. "
        f"현재 대화에서 다루는 도메인과 사용 가능한 슬롯은 다음과 같습니다: "
        f"{json.dumps(available_slots, indent=2, ensure_ascii=False)}. "
        "각 도메인에 대해 자연스러운 대화와 적절한 슬롯-값 쌍을 생성하세요. "
        "모든 대화는 반드시 한국어로 생성해야 하며, 실제 사람들이 대화하는 것처럼 자연스러워야 합니다. "
        "시스템과 사용자 간의 대화가 자연스러운 한국어로 이루어져야 합니다."
        "\n\n대화 규칙:"
        "\n1. 첫 턴(turn_id=1)에서만 시스템 응답이 빈 문자열('')이어야 합니다."
        "\n2. 그 외의 턴에서는 시스템이 사용자의 요청에 대해 상세하고 친절한 응답을 제공해야 합니다."
        "\n3. 사용자는 반드시 반말(informal speech)을 사용해야 합니다."
        "\n4. 사용자의 발화는 단순한 명령이 아닌, 자연스러운 대화체로 작성해주세요."
        "\n5. 시스템은 항상 존댓말을 사용합니다."
        "\n6. 시스템 응답은 단순히 작업 수행 결과만 알리지 말고, 추가 도움이 필요한지 등을 포함해야 합니다."
    )

    if is_last_turn:
        base_prompt += (
            "\n\n현재는 대화의 마지막 턴입니다. 사용자는 대화를 마무리하는 발화를 해야 합니다."
            "\n마지막 발화 예시:"
            "\n- '그래, 이제 다 됐어. 고마워.'"
            "\n- '그게 다야. 도와줘서 고마워.'"
            "\n- '괜찮아. 더 필요한 건 없어.'"
            "\n- '그래, 다 했어. 수고했어.'"
        )

    if previous_turn:
        context = (
            "\n이전 턴의 대화 내용입니다:"
            f"\n시스템: {previous_turn['dialog']['sys'][-1]}"
            f"\n사용자: {previous_turn['dialog']['usr'][-1]}"
            f"\n이전 턴의 슬롯 값: {json.dumps(previous_turn['turn_slot_values'], indent=2, ensure_ascii=False)}"
            f"\n현재 대화에서 다뤄야 할 도메인: {', '.join(domains)}"
        )
        base_prompt += context

    return base_prompt


def generate_dialog(client: AzureOpenAI, num_examples=10):
    dialogues = []

    for dialog_idx in range(1, num_examples + 1):
        dialog_id = f"dialog_{dialog_idx}"
        num_domains = random.randint(1, 2)
        selected_domains = random.sample(list(DOMAIN_SLOTS.keys()), num_domains)

        turns_per_dialog = random.randint(3, 5)

        current_dialog = []
        accumulated_slot_values = []

        for turn_idx in range(1, turns_per_dialog + 1):
            start_time = time.time()
            previous_turn = current_dialog[-1] if current_dialog else None
            is_last_turn = turn_idx == turns_per_dialog
            system_prompt = create_system_prompt(
                selected_domains, previous_turn, is_last_turn
            )

            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"다음 도메인에 대한 대화의 턴 {turn_idx}를 생성하세요: {selected_domains}. "
                            "이전 대화 문맥을 고려하여 자연스러운 대화를 이어가세요."
                            f"{'현재가 첫 번째 턴이므로, 시스템은 빈 문자열('')로 응답해야 합니다.' if is_last_turn else ''}"
                            f"{'현재가 마지막 턴이므로, 사용자가 대화를 마무리하는 것으로 끝내주세요.' if is_last_turn else ''}"
                        ),
                    },
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format=TurnData,
            )
            if completion:
                turn_data = completion.choices[0].message.parsed

                if turn_idx == 1:
                    turn_data.dialog.sys = [""]

                turn_data.turn_slot_values = [
                    SlotValue(**slot) if isinstance(slot, dict) else slot
                    for slot in turn_data.turn_slot_values
                ]
                turn_data.last_slot_values = [
                    SlotValue(**slot) if isinstance(slot, dict) else slot
                    for slot in (turn_data.last_slot_values or [])
                ]

                turn_data.ID = dialog_id
                turn_data.turn_id = turn_idx
                turn_data.domains = selected_domains

                if previous_turn:
                    turn_data.last_slot_values = [
                        SlotValue(**slot) if isinstance(slot, dict) else slot
                        for slot in previous_turn["turn_slot_values"]
                    ]
                    accumulated_slot_values.extend(turn_data.last_slot_values)

                turn_data.slot_values = (
                    accumulated_slot_values + turn_data.turn_slot_values
                )
                current_dialog.append(turn_data.model_dump())
                elapsed_time = time.time() - start_time
                print(
                    f"✅ [{dialog_idx}/{num_examples}] 대화 {turn_idx}/{turns_per_dialog} 생성 완료: "
                    f"{selected_domains} (⏱️ {elapsed_time:.2f}초)"
                )
        dialogues.extend(current_dialog)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)


def main():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = "2024-08-01-preview"

    if not api_key or not azure_endpoint:
        raise ValueError(
            "환경 변수 'AZURE_OPENAI_API_KEY'와 'AZURE_OPENAI_ENDPOINT'를 설정해야 합니다."
        )

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    try:
        print("=" * 100)
        print("💾 데이터를 저장 중")
        start_time = time.time()
        generate_dialog(client, num_examples=10)
        elapsed_time = time.time() - start_time
        print(f"🎉 데이터 생성 완료: 총 걸린 시간 : {elapsed_time}")
        print("=" * 100)
    except Exception as e:
        print(f"데이터 생성 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
