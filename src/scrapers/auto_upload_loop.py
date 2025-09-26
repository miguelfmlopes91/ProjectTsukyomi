import time
import sys
import traceback
from upload_to_gsheet import main

def run_forever(interval=5):
    print(f"🚀 Starting loop (every {interval}s, Ctrl+C to stop)...")
    while True:
        try:
            print(f"\n🕒 {time.strftime('%H:%M:%S')} updating...")
            main()  # gera/atualiza "live - Timeline.csv"
        except Exception:
            print("❌ Error occurred:")
            traceback.print_exc()
            time.sleep(2)  # pequeno backoff em caso de erro
        time.sleep(interval)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_forever(interval)
