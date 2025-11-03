
import os, math, time, numpy as np

try:
    import mujoco
    import glfw
except Exception as e:
    raise RuntimeError(f"RobotMotionViewer requires mujoco+glfw installed: {e}")

def _try_paths(candidates):
    last_err = None
    for p in candidates:
        if not p: 
            continue
        if os.path.exists(p):
            try:
                return mujoco.MjModel.from_xml_path(p)
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    return None

class RobotMotionViewer:
    """
    MuJoCo viewer that applies root translation/orientation + joint positions.
    If model_path is given, it will be used directly. Otherwise we search a set of common locations.
    """
    def __init__(self, robot="g1", camera_follow=True, motion_fps=30,
                 transparent_robot=0, record_video=False, dt=None, ground=True,
                 model_path=None):
        # 1) Direct path wins (if provided)
        search = []
        if model_path:
            search.append(model_path)

        # 2) Env var override (handy for Jak's setups)
        env_model = os.getenv("GMR_G1_XML") or os.getenv("GMR_ROBOT_XML")
        if env_model:
            search.append(env_model)

        # 3) Common repo-relative locations (Windows-friendly)
        common = [
            # GMR assets (as shown in your earlier logs)
            os.path.join("assets", "unitree_g1", "g1_mocap_29dof.xml"),
            os.path.join("assets", "unitree_g1", "g1_29dof_rev_1_0.xml"),
            # robot_description style
            os.path.join("robot_description", "g1", "g1_mocap_29dof.xml"),
            os.path.join("robot_description", "g1", "g1_29dof_rev_1_0.xml"),
            os.path.join("robot_description", "g1", "g1_29dof_rev_1_0.urdf"),
        ]
        search.extend(common)

        # 4) Try with repo root prefixes: . and jak_experiment parent
        #    so calls from subfolders still find them
        prefixes = [".", os.path.join(".."), os.path.join("..", "..")]
        expanded = []
        for pre in prefixes:
            for c in common:
                expanded.append(os.path.join(pre, c))
        search.extend(expanded)

        # Resolve model
        model = _try_paths(search)
        if model is None:
            raise FileNotFoundError(f"Could not load robot XML/URDF. Tried:\n  " + "\n  ".join(search))

        self.model = model
        self.data = mujoco.MjData(self.model)

        if self.model.nq < 7:
            raise RuntimeError("Model has fewer than 7 qpos -> no free-flyer root.")

        self.camera_follow = camera_follow
        self.motion_fps = motion_fps
        self.dt = (1.0/motion_fps) if dt is None else dt

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.window = glfw.create_window(1280, 800, "RobotMotionViewer", None, None)
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.azimuth = 140.0
        self.cam.elevation = -20.0
        self.cam.distance = 4.0
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    def step(self, root_pos, root_quat_wxyz, qpos_tail):
        if glfw.window_should_close(self.window):
            return

        root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=float)
        root_quat_wxyz = root_quat_wxyz / (np.linalg.norm(root_quat_wxyz) + 1e-12)

        self.data.qpos[0:3] = np.asarray(root_pos, dtype=float)
        self.data.qpos[3:7] = root_quat_wxyz
        if qpos_tail is not None and len(qpos_tail) > 0:
            n_tail = min(self.model.nq - 7, len(qpos_tail))
            self.data.qpos[7:7+n_tail] = qpos_tail[:n_tail]

        mujoco.mj_forward(self.model, self.data)

        if self.camera_follow:
            self.cam.lookat = self.data.qpos[0:3].copy()
            self.cam.distance = 3.5

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        time.sleep(self.dt)

    def close(self):
        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass
