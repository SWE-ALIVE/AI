import os
from openai import OpenAI
from typing import Literal
import json
import time
import random
from pydantic import BaseModel


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


domain_slot_file = "app/config/domain_slots.json"
output_file = "app/data/gpt4o_dataset.json"
num_examples = 240

with open(domain_slot_file, "r") as f:
    DOMAIN_SLOTS = json.load(f)

DOMAIN_TYPES = Literal[tuple(DOMAIN_SLOTS.keys())]
SLOT_TYPES = {domain: Literal[tuple(slots)] for domain, slots in DOMAIN_SLOTS.items()}


last_turn_prompt = (
    "\n\n현재는 대화의 마지막 턴입니다. 사용자는 반드시 대화를 마무리하는 내용을 발화에 포함시켜야 합니다."
    "\n마지막 발화 예시:"
    "\n- '1. 그래, 이제 다 됐어. 고마워.'"
    "\n- '2. 그게 다야. 도와줘서 고마워.'"
    "\n- '3. 괜찮아. 더 필요한 건 없어.'"
    "\n- '4.그래, 다 했어. 수고했어.'"
    "\n- '5. 이제 그만.'"
)


def create_system_prompt(
    domains: list[str], previous_turn: dict = None, is_last_turn: bool = False
):
    available_slots = {domain: DOMAIN_SLOTS[domain] for domain in domains}
    slot_constraints = {}

    for domain, slots in available_slots.items():
        slot_constraints[domain] = {}
        for slot, values in slots.items():
            if isinstance(values, dict) and values.get("type") == "range":
                slot_constraints[domain][slot] = f"{values['min']} ~ {values['max']}"
            elif isinstance(values, list):
                slot_constraints[domain][slot] = ", ".join(values)

    base_prompt = (
        "당신은 다중 도메인 작업 지향 대화를 생성하는 AI입니다. "
        "각 도메인(가전제품)에 대해 자연스러운 대화와 적절한 슬롯-값 쌍을 생성하세요. "
        "모든 대화는 반드시 한국어로 생성해야 하며, 실제 사람들이 대화하는 것처럼 자연스러워야 합니다. "
        "시스템과 사용자 간의 대화가 자연스러운 한국어로 이루어져야 합니다."
        "스키마 설명입니다:\n"
        "- ID: 대화 식별자 (예: 'ALIVE0000')\n"
        "- turn_id: 현재 대화의 턴 번호 (0부터 시작)\n"
        "- domains: 현재 대화에 참여하는 가전제품 목록\n"
        "- dialog: 대화 내용\n"
        "  - sys: 시스템 발화 목록\n"
        "  - usr: 사용자 발화 목록\n"
        "- slot_values: 대화 시작부터 현재까지 누적된 모든 슬롯 값들\n"
        "- turn_slot_values: 현재 턴에서 변경된 슬롯 값들\n"
        "- last_slot_values: 이전 턴에서 변경된 슬롯 값들\n\n"
        "슬롯 값 기록 시 주의사항:\n"
        "1. turn_slot_values는 현재 턴에서 사용자의 발화에 의해 변경된 값만 포함\n"
        "2. 이전 턴의 값은 last_slot_values에 자동으로 포함됨\n"
        "3. slot_values는 last_slot_values와 turn_slot_values의 누적 리스트\n\n"
        "\n\n대화 규칙:"
        "\n1. 첫 턴(turn_id=0)에서만 시스템 응답이 빈 문자열('')이어야 합니다. 그 외의 턴에서는 시스템이 사용자의 요청에 대해 상세하고 친절한 응답을 제공해야 합니다."
        "\n2. 시스템은 반드시 존댓말을 사용해야 합니다."
        "\n3. 사용자는 반드시 반말(informal speech)을 사용해야 합니다."
        "\n4. 사용자의 발화는 단순한 명령이 아닌, 자연스러운 대화체로 작성해주세요."
        f"현재 대화에서 다루는 도메인과 사용 가능한 슬롯은 다음과 같습니다: "
        f"{json.dumps(slot_constraints, indent=2, ensure_ascii=False)}. "
    )

    if is_last_turn:
        base_prompt += last_turn_prompt

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


def generate_dialog(client: OpenAI, num_examples=10):
    dialogues = []

    for dialog_idx in range(0, num_examples):
        num_domains = random.randint(1, 2)
        selected_domains = random.sample(list(DOMAIN_SLOTS.keys()), num_domains)

        turns_per_dialog = random.randint(3, 5)

        current_dialog = []

        for turn_idx in range(0, turns_per_dialog):
            start_time = time.time()
            previous_turn = current_dialog[-1] if current_dialog else None
            is_last_turn = turn_idx == turns_per_dialog - 1
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
                            f"{last_turn_prompt if is_last_turn else ''}"
                        ),
                    },
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format=TurnData,
            )
            if completion:
                turn_data = completion.choices[0].message.parsed

                if turn_idx == 0:
                    turn_data.dialog.sys = [""]

                current_dialog.append(turn_data.model_dump())
                elapsed_time = time.time() - start_time
                print(
                    f"✅ [{dialog_idx + 1}/{num_examples}] 대화 {turn_idx + 1}/{turns_per_dialog} 생성 완료: "
                    f"{selected_domains} (⏱️ {elapsed_time:.2f}초)"
                )
        dialogues.extend(current_dialog)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수 'OPENAI_API_KEY'를 설정해야 합니다.")
    client = OpenAI(api_key=api_key)
    try:
        print("=" * 100)
        print("💾 데이터를 저장 중")
        start_time = time.time()
        generate_dialog(client, num_examples=num_examples)
        elapsed_time = time.time() - start_time
        print(f"🎉 데이터 생성 완료: 총 걸린 시간 : {elapsed_time:.2f}초")
        print("=" * 100)
    except Exception as e:
        print(f"데이터 생성 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
