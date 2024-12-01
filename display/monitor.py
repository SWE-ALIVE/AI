# app/monitor.py
from status_viewer import DeviceStatusViewer
import time
from pathlib import Path


def main():
    viewer = DeviceStatusViewer()
    last_modified = None
    state_file = Path("app/config/device_states.json")

    print("가전제품 상태 모니터를 시작합니다.")
    print("종료하려면 Ctrl+C를 누르세요.")

    try:
        while True:
            if state_file.exists():
                current_modified = state_file.stat().st_mtime
                # 파일이 수정되었거나 처음 실행할 때
                if last_modified != current_modified:
                    viewer.show_status()
                    last_modified = current_modified

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
