# status_viewer.py
import json
import time
from pathlib import Path
import os


class DeviceStatusViewer:
    def __init__(self, state_file="app/config/device_states.json"):
        self.state_file = Path(state_file)
        self.default_states = {
            "AIR_CONDITIONER": {
                "power": "꺼짐",
                "temperature": 24,
                "mode": "자동",
                "fan_speed": "자동",
            },
            "AIR_PURIFIER": {"power": "꺼짐", "mode": "자동", "fan_speed": "자동"},
            "TV": {"power": "꺼짐", "volume": 0, "channel": 1, "mode": "일반"},
            "ROBOT_CLEANER": {"power": "꺼짐", "mode": "보통"},
        }

    def get_state(self, device_type, current_states):
        try:
            state = self.default_states[device_type].copy()
            if device_type in current_states:
                for key, value in current_states[device_type].items():
                    key = key.lower()
                    if key in state:
                        state[key] = value
            return state
        except Exception as e:
            print(f"Error getting state for {device_type}: {e}")
            return self.default_states[device_type]

    def get_device_art(self, device_type, state):
        try:
            # ANSI 색상 코드
            RESET = "\033[0m"
            RED = "\033[31m"
            GRAY = "\033[90m"

            # 선 색상: 켜짐이면 빨간색, 꺼짐이면 회색
            color = RED if state["power"] == "켜짐" else GRAY

            if device_type == "AIR_CONDITIONER":
                return f"""
      {color}╔════════════════════╗{RESET}
      {color}║     에어컨         ║{RESET}
      {color}╠════════════════════╣{RESET}
      {color}║ 온도: {state['temperature']}°C         ║{RESET}
      {color}║ 모드: {state['mode']}         ║{RESET}
      {color}  풍량: {state['fan_speed']}          {RESET}
      {color}╚════════════════════╝{RESET}
      """
            elif device_type == "AIR_PURIFIER":
                return f"""
      {color}╔════════════════════╗{RESET}
      {color}║   공기청정기       ║{RESET}
      {color}╠════════════════════╣{RESET}
      {color}║ 모드: {state['mode']}         ║{RESET}
      {color}║ 풍량: {state['fan_speed']}         ║{RESET}
      {color}                       {RESET}
      {color}╚════════════════════╝{RESET}
      """
            elif device_type == "TV":
                return f"""
      {color}╔══════════════════╗{RESET}
      {color}║       TV         ║{RESET}
      {color}╠══════════════════╣{RESET}
      {color}║ 볼륨: {str(state['volume']).rjust(3)}        ║{RESET}
      {color}║ 채널: {str(state['channel']).rjust(3)}        ║{RESET}
      {color} 모드: {state['mode']}        {RESET}
      {color}╚══════════════════╝{RESET}
      """
            elif device_type == "ROBOT_CLEANER":
                return f"""
      {color}╔════════════════════╗{RESET}
      {color}║   로봇청소기       ║{RESET}
      {color}╠════════════════════╣{RESET}
      {color}║ 모드: {state['mode']}         ║{RESET}
      {color}║          {RESET}
      {color}          {RESET}
      {color}╚════════════════════╝{RESET}
      """
            else:
                return "Unknown device"
        except Exception as e:
            print(f"Error generating art for {device_type}: {e}")
            return f"Error displaying {device_type}"

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_status(self):
        self.clear_screen()
        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as f:
                    current_states = json.load(f)
            else:
                current_states = {}

            # 모든 디바이스의 ASCII 아트 생성
            arts = []
            for device in self.default_states.keys():
                state = self.get_state(device, current_states)
                art = self.get_device_art(device, state)
                arts.append(art)

            art_lines = [art.split("\n") for art in arts]
            max_lines = max(len(lines) for lines in art_lines)

            for i in range(max_lines):
                line = ""
                for idx, device_lines in enumerate(art_lines):
                    if i < len(device_lines):
                        line += device_lines[i].ljust(25)  # 간격 늘림
                    else:
                        line += " " * 25
                    if idx == 1 and i == len(device_lines) - 1:
                        line += "\n"  # 줄바꿈 추가
                print(line)

            # 시간 표시 추가
            print("\n" + "=" * 100)
            print(f"마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"Error displaying status: {e}")
            print("Using default states...")
            time.sleep(1)
