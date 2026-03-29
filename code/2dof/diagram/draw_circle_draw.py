from pathlib import Path
import math
import time
import numpy as np
import mujoco
import mujoco.viewer

BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR.parent / "two_link_drawer.xml"

Q1_MIN, Q1_MAX = -2.2, 2.2
Q2_MIN, Q2_MAX = 0.1, 2.6

Kp = 5.0
DT = 0.01

DRAW_START_TIME = 0.5
MAX_MARKERS = 800


def circle_target(t, center=(0.55, 0.15), radius=0.08, period=6.0):
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


def add_marker(scene, pos, size=0.02, rgba=(0.1, 0.1, 0.1, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, size, size], dtype=np.float64),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).reshape(-1),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    print("draw_circle_copy 디버그 실행중")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    ee_id = model.site("ee").id
    pen_tip_id = model.site("pen_tip").id

    trace_ee = []
    trace_pen = []

    data.qpos[0] = 0.3
    data.qpos[1] = 1.0
    clamp_q(data.qpos)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as v:
        start_wall = time.time()

        while v.is_running():
            t = data.time

            p_des, v_des = circle_target(t)

            ee_pos = data.site_xpos[ee_id].copy()
            p_cur = ee_pos[:2]

            v_cmd = v_des + Kp * (p_des - p_cur)

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)

            J = jacp[:2, :2]
            qdot = np.linalg.pinv(J) @ v_cmd

            data.qpos[0] += qdot[0] * DT
            data.qpos[1] += qdot[1] * DT
            clamp_q(data.qpos)

            data.qvel[0] = qdot[0]
            data.qvel[1] = qdot[1]

            mujoco.mj_forward(model, data)

            ee_pos = data.site_xpos[ee_id].copy()
            pen_pos = data.site_xpos[pen_tip_id].copy()

            if data.time > DRAW_START_TIME:
                trace_ee.append((ee_pos[0], ee_pos[1]))
                trace_pen.append((pen_pos[0], pen_pos[1]))

            if len(trace_ee) % 30 == 0 and len(trace_ee) > 0:
                print("trace_ee:", len(trace_ee), "trace_pen:", len(trace_pen))
                print("ee xyz =", ee_pos)
                print("pen xyz =", pen_pos)

            with v.lock():
                v.user_scn.ngeom = 0

                for x, y in trace_pen[-MAX_MARKERS:]:
                    add_marker(
                        v.user_scn,
                        [x, y, 0.02],
                        size=0.005,
                        rgba=(0.0, 0.0, 0.0, 1.0),
                    )

                add_marker(
                    v.user_scn,
                    [p_des[0], p_des[1], 0.03],
                    size=0.008,
                    rgba=(1.0, 0.0, 0.0, 1.0),
                )

            v.sync()

            elapsed = time.time() - start_wall
            delay = DT - (elapsed - data.time)
            if delay > 0:
                time.sleep(delay)

            data.time += DT


if __name__ == "__main__":
    main()