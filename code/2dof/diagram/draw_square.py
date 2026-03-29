from pathlib import Path
import math
import time
import numpy as np
import mujoco
import mujoco.viewer

BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR.parent / "two_link_drawer.xml"

# 관절 범위
Q1_MIN, Q1_MAX = -2.2, 2.2
Q2_MIN, Q2_MAX = 0.1, 2.6

# 추적 파라미터
Kp = 5.0
DT = 0.01

# 종이 높이
PAPER_Z = -0.12
PAPER_TOL = 0.01


def square_target(t, center=(0.55, 0.15), size=0.20, period=7.0):
    cx, cy = center
    h = size / 2.0
    phase = (t % period) / period

    edge_time = period * 0.25
    speed = (2.0 * h) / edge_time

    if phase < 0.25:
        a = phase / 0.25
        p = np.array([cx - h + 2*h*a, cy - h])
        v = np.array([speed, 0.0])
    elif phase < 0.50:
        a = (phase - 0.25) / 0.25
        p = np.array([cx + h, cy - h + 2*h*a])
        v = np.array([0.0, speed])
    elif phase < 0.75:
        a = (phase - 0.50) / 0.25
        p = np.array([cx + h - 2*h*a, cy + h])
        v = np.array([-speed, 0.0])
    else:
        a = (phase - 0.75) / 0.25
        p = np.array([cx - h, cy + h - 2*h*a])
        v = np.array([0.0, -speed])

    return p, v


def clamp_q(q):
    q[0] = max(Q1_MIN, min(Q1_MAX, q[0]))
    q[1] = max(Q2_MIN, min(Q2_MAX, q[1]))
    return q


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    pen_tip_id = model.site("pen_tip").id

    trace = []
    target_trace = []

    # 시작점을 네모의 첫 점으로 맞춤
    p0, _ = square_target(0.0)
    data.qpos[0] = 0.3
    data.qpos[1] = 1.0
    clamp_q(data.qpos)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.time()

        while viewer.is_running():
            t = data.time

            # 목표 네모 위치, 속도
            p_des, v_des = square_target(t)
            target_trace.append((p_des[0], p_des[1]))

            # 현재 펜 위치
            pen_pos = data.site_xpos[pen_tip_id].copy()
            p_cur = pen_pos[:2]

            # 위치 오차 포함 목표 속도
            v_cmd = v_des + Kp * (p_des - p_cur)

            # Jacobian 계산
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, pen_tip_id)

            J = jacp[:2, :2]

            # pseudoinverse
            qdot = np.linalg.pinv(J) @ v_cmd

            # 적분
            data.qpos[0] += qdot[0] * DT
            data.qpos[1] += qdot[1] * DT
            clamp_q(data.qpos)

            data.qvel[0] = qdot[0]
            data.qvel[1] = qdot[1]

            mujoco.mj_forward(model, data)

            pen_pos = data.site_xpos[pen_tip_id].copy()
            if data.time > 1.2 and abs(pen_pos[2] - PAPER_Z) < PAPER_TOL:
                trace.append((pen_pos[0], pen_pos[1]))

            if len(target_trace) % 100 == 0:
                err = np.linalg.norm(p_des - pen_pos[:2])
                print(f"t={t:.2f} err={err:.4f}")

            viewer.sync()

            elapsed = time.time() - start_wall
            delay = DT - (elapsed - data.time)
            if delay > 0:
                time.sleep(delay)

            data.time += DT

    np.save("ee_trace.npy", np.array(trace))
    np.save("target_trace.npy", np.array(target_trace))
    print("저장 완료: ee_trace.npy, target_trace.npy")


if __name__ == "__main__":
    main()