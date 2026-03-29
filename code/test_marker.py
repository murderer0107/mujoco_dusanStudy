import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "two_link_drawer.xml"

def add_big_red_marker(scene, pos):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.08, 0.08, 0.08], dtype=np.float64),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).reshape(-1),
        rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 그냥 정면 쪽에 보일 만한 좌표
    marker_pos = [0.6, 0.0, 0.15]

    with mujoco.viewer.launch_passive(model, data) as v:
        start = time.time()
        while v.is_running() and time.time() - start < 20:
            mujoco.mj_forward(model, data)

            with v.lock():
                v.user_scn.ngeom = 0
                add_big_red_marker(v.user_scn, marker_pos)

            v.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()