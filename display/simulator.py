# simulate.py
import json
import time
from pathlib import Path
import random
from state_manager import DeviceStateManager
from status_viewer import DeviceStatusViewer


def generate_mock_context():
    """테스트용 컨텍스트 생성"""
    devices = ["AIR_CONDITIONER", "AIR_PURIFIER", "TV", "ROBOT_CLEANER"]
    device = random.choice(devices)

    updates = {
        "AIR_CONDITIONER": {
            "power": ["켜짐", "꺼짐"],
            "temperature": range(16, 31),
            "mode": ["냉방", "난방", "자동", "제습"],
            "fan_speed": ["약함", "보통", "강함", "자동"],
        },
        "AIR_PURIFIER": {
            "power": ["켜짐", "꺼짐"],
            "mode": ["자동", "수면", "터보"],
            "fan_speed": ["약함", "보통", "강함", "자동"],
        },
        "TV": {
            "power": ["켜짐", "꺼짐"],
            "volume": range(0, 101),
            "channel": range(1, 1000),
            "mode": ["일반", "영화", "스포츠", "게임"],
        },
        "ROBOT_CLEANER": {
            "power": ["켜짐", "꺼짐"],
            "mode": ["지그재그", "꼼꼼", "집중", "보통"],
        },
    }

    # 랜덤하게 1-2개의 슬롯 선택
    slots = list(updates[device].keys())
    selected_slots = random.sample(slots, random.randint(1, 2))

    context = []
    for slot in selected_slots:
        value = (
            random.choice(updates[device][slot])
            if isinstance(updates[device][slot], list)
            else random.choice(list(updates[device][slot]))
        )
        context.append({"domain": device, "slot": slot, "value": value})

    return context


def main():
    state_manager = DeviceStateManager()

    print("Starting device simulator...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            # 랜덤 업데이트 생성 및 적용
            context = generate_mock_context()
            state_manager.update_states(context)

            # 마지막 업데이트 내용 표시
            print("\nLast update:")
            for update in context:
                print(f"{update['domain']}: {update['slot']} -> {update['value']}")

            # 3초 대기
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
