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


def circle_target(t, center=(0.55, 0.15), radius=0.12, period=6.0):
    w = 2.0 * math.pi / period
    cx, cy = center
    x = cx + radius * math.cos(w * t)
    y = cy + radius * math.sin(w * t)
    vx = -radius * w * math.sin(w * t)
    vy =  radius * w * math.cos(w * t)
    return np.array([x, y]), np.array([vx, vy])


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

    # 시작점을 원의 첫 점으로 맞춤
    p0, _ = circle_target(0.0)
    data.qpos[0] = 0.3
    data.qpos[1] = 1.0
    clamp_q(data.qpos)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.time()

        while viewer.is_running():
            t = data.time

            # 목표 원 위치, 속도
            p_des, v_des = circle_target(t)
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

            J = jacp[:2, :2]   # x,y 와 joint1,joint2만 사용

            # pseudoinverse
            qdot = np.linalg.pinv(J) @ v_cmd

            # 적분
            data.qpos[0] += qdot[0] * DT
            data.qpos[1] += qdot[1] * DT
            clamp_q(data.qpos)

            # 속도도 맞춰주면 좀 더 부드러움
            data.qvel[0] = qdot[0]
            data.qvel[1] = qdot[1]

            mujoco.mj_forward(model, data)

            # 종이에 닿았을 때만 기록
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