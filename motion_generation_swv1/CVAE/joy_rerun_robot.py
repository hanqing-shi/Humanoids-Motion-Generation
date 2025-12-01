#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import rerun as rr
import hid

# ==============================
# PATH / CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "retarget_out_vel", "walk1_subject1_mid_1_vel.csv")

FPS = 30.0              # 시뮬레이션 프레임레이트
DT = 1.0 / FPS          # 타임스텝
MAX_LIN_VEL = 1.0       # [m/s] 정도로 가정 (원하는 값으로 조정)
MAX_ANG_VEL = 1.0       # [rad/s]

# Logitech Cordless RumblePad 2
VENDOR_ID = 0x046D
PRODUCT_ID = 0xC219


# ==============================
# JOYSTICK (HID) FUNCTIONS
# ==============================

def open_rumblepad():
    dev = hid.device()
    dev.open(VENDOR_ID, PRODUCT_ID)
    dev.set_nonblocking(True)
    return dev

def _norm_axis(v: int) -> float:
    """0~255 -> -1.0~+1.0 범위로 정규화"""
    return (v - 128) / 128.0

def decode_rumblepad(report):
    """
    HID raw report(리스트/바이트) -> 축/버튼 값 디코딩
    report 길이는 보통 7 이상 (0~6 사용)
    """
    if len(report) < 7:
        return None

    buttons_low  = report[0]
    buttons_high = report[1]

    buttons = []
    for i in range(8):
        if buttons_low & (1 << i):
            buttons.append(i)
    for i in range(4):
        if buttons_high & (1 << i):
            buttons.append(i + 8)

    lx = _norm_axis(report[2])
    ly = _norm_axis(report[3])
    rx = _norm_axis(report[4])
    ry = _norm_axis(report[5])
    dpad = report[6]

    return {
        "buttons": buttons,
        "axes": {
            "lx": lx,
            "ly": ly,
            "rx": rx,
            "ry": ry,
        },
        "dpad": dpad,
    }


# ==============================
# RERUN + ROBOT STATE
# ==============================

def init_robot_state_from_csv(csv_path):
    """
    너가 기존에 쓰던 CSV에서 q_t, vel_t 뽑던 것 그대로 사용.
    여기서는 q_t만 사용해서 초기 상태로 사용.
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    # D_CONF가 코드 밖에 있으면 CSV 형태 보고 직접 잘라줘도 됨.
    # 예: 앞 36개가 q라면:
    # q_t = data[0, :36]
    # 여기서는 그냥 앞 36이라고 가정 (원하는 대로 수정)
    D_CONF = 36
    q_t = data[0, :D_CONF]
    return q_t.copy()      # (D_CONF,) numpy array

def init_rerun():
    rr.init("Joystick Controlled Robot", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def log_robot_to_rerun(q, frame_idx):
    """
    q: (D_CONF,) 벡터
    여기에서는 가장 앞 3개를 base position (x,y,z)라고 가정해서 point로 찍어줌.
    나중에 rerun_urdf 연결하면 여기서 URDF link들 transform 찍으면 됨.
    """
    rr.set_time_sequence("frame", frame_idx)

    base_xyz = q[:3]  # [x, y, z] 로 가정
    rr.log(
        "robot/base",
        rr.Points3D(
            positions=np.array([base_xyz]),
            radii=0.03,
            colors=np.array([[0, 255, 0]], dtype=np.uint8),  # 초록색
        ),
    )

    # 디버깅용으로 전체 q를 벡터로 찍고 싶으면:
    rr.log("robot/q", rr.Tensor(q.astype(np.float32)))


# ==============================
# JOYSTICK -> VELOCITY MAPPING
# ==============================

def joystick_to_velocities(decoded):
    """
    조이스틱 축 값을 (vx, vy, wz) 같은 로봇 base 속도로 매핑.
    여기선 예시:
      - 왼스틱 X: 좌/우 (vy)
      - 왼스틱 Y: 앞/뒤 (vx)
      - 오른스틱 X: yaw 속도 (wz)
    """
    if decoded is None:
        return 0.0, 0.0, 0.0

    axes = decoded["axes"]

    # ly는 위로 밀면 -값이라 부호 반전해서 앞 방향을 +로 맞춤
    vx = -axes["ly"] * MAX_LIN_VEL
    vy =  axes["lx"] * MAX_LIN_VEL
    wz =  axes["rx"] * MAX_ANG_VEL

    return vx, vy, wz


# ==============================
# MAIN LOOP
# ==============================

def main():
    # 1) Rerun & 로봇 초기 상태
    init_rerun()
    q = init_robot_state_from_csv(CSV_PATH)  # shape: (D_CONF,)
    D_CONF = q.shape[0]

    # base pose를 q[0:3], yaw를 q[5] 같은 식으로 쓰고 있다면
    # 아래 매핑도 그에 맞게 바꿔주면 됨.
    # 여기서는 간단히:
    #   q[0] = x, q[1] = y, q[2] = z, q[5] = yaw  (예시)
    # 라고 가정
    base_idx_x = 0
    base_idx_y = 1
    base_idx_z = 2
    yaw_idx    = 5 if D_CONF > 5 else 2  # 없으면 z에다 대충…

    # 2) 조이스틱
    dev = open_rumblepad()
    print("✅ RumblePad connected. Move sticks to control the robot.")

    frame_idx = 0
    t = 0.0

    try:
        while True:
            report = dev.read(64)
            decoded = decode_rumblepad(report) if report else None

            # 3) 조이스틱 입력 -> 속도
            vx, vy, wz = joystick_to_velocities(decoded)

            # 4) 심플한 2D 평면 상에서 base x,y,yaw 적분 (z는 고정)
            #    q[base_idx_x], q[base_idx_y], q[yaw_idx] 만 갱신
            # 현재 yaw
            yaw = float(q[yaw_idx])

            # world frame 이동량 (vx,vy가 body frame 기준이라면 회전 변환 해줄 수도 있음)
            # 여기서는 간단하게 world frame에 바로 적용
            q[base_idx_x] += vx * DT
            q[base_idx_y] += vy * DT
            q[yaw_idx]    += wz * DT

            # yaw wrap
            q[yaw_idx] = (q[yaw_idx] + math.pi) % (2 * math.pi) - math.pi

            # 5) Rerun 로깅
            log_robot_to_rerun(q, frame_idx)

            frame_idx += 1
            t += DT
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\n[joy_rerun_robot] 종료합니다.")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
