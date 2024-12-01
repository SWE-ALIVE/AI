# app/state_manager.py
import json
from pathlib import Path


class DeviceStateManager:
    def __init__(self, initial_state_file="app/config/initial_states.json"):
        self.state_file = Path("app/config/device_states.json")
        self.initial_state_file = Path(initial_state_file)
        self.initialize_states()

    def initialize_states(self):
        print("Initializing device states...")
        initial_states = {
            "AIR_CONDITIONER": {
                "power": "꺼짐",
                "temperature": 24,
                "mode": "자동",
                "fan_speed": "자동",
            },
            "AIR_PURIFIER": {"power": "꺼짐", "mode": "자동", "fan_speed": "자동"},
            "TV": {"power": "꺼짐", "volume": 30, "channel": 1, "mode": "일반"},
            "ROBOT_CLEANER": {"power": "꺼짐", "mode": "보통"},
        }

        # 초기 상태 저장
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(initial_states, f, ensure_ascii=False, indent=2)

    def update_states(self, context_updates):
        """context 업데이트를 사용하여 상태 업데이트"""
        with open(self.state_file, "r", encoding="utf-8") as f:
            current_states = json.load(f)

        # context의 각 업데이트 적용
        for update in context_updates:
            domain = update["domain"]
            slot = update["slot"]
            value = update["value"]

            # 상태 업데이트
            if domain in current_states:
                current_states[domain][slot] = value

        # 업데이트된 상태 저장
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(current_states, f, ensure_ascii=False, indent=2)
