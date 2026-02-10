import hid
import time

# Logitech Cordless RumblePad 2
VENDOR_ID = 0x046D
PRODUCT_ID = 0xC219   # Cordless RumblePad 2 PID

def list_devices():
    print("=== HID 디바이스 목록 ===")
    for d in hid.enumerate():
        vid = d["vendor_id"]
        pid = d["product_id"]
        name = d.get("product_string", "")
        mfg  = d.get("manufacturer_string", "")
        print(f"VID: {vid:04X}, PID: {pid:04X}, 제조사: {mfg}, 이름: {name}")

def open_rumblepad():
    dev = hid.device()
    dev.open(VENDOR_ID, PRODUCT_ID)
    # non-blocking 모드: 데이터 없으면 바로 리턴
    dev.set_nonblocking(True)
    return dev

def main():
    list_devices()

    print("\nLogitech Cordless RumblePad 2 열기 시도 중...")
    try:
        dev = open_rumblepad()
    except Exception as e:
        print(f"❌ 열기 실패: {e}")
        print(" - USB 리시버가 꽂혀 있는지, 패드 전원이 켜져 있는지 확인")
        return

    print("✅ RumblePad 열기 성공!")
    print("버튼/스틱을 움직이면 Raw HID 데이터가 찍힙니다.")
    print("종료: Ctrl + C\n")

    try:
        while True:
            # 64바이트 정도 버퍼로 읽기 (기기마다 길이는 다를 수 있음)
            data = dev.read(64)
            if data:
                # 0만 가득 들어오는 패킷은 버려도 됨 (필요하면 주석 처리)
                if any(b != 0 for b in data):
                    print("입력 데이터:", data)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        dev.close()

if __name__ == "__main__":
    main()
