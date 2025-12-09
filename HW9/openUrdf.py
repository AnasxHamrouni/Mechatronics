import os
import mujoco
import mujoco.viewer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "ur_description", "urdf", "ur3e.urdf")
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return
    print(f"Loading URDF from: {urdf_path}")
    mesh_path = os.path.join(
        script_dir, "ur_description", "meshes", "ur3e", "collision", "base.stl"
    )
    print("Mesh exists:", os.path.exists(mesh_path), mesh_path)
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")

    if model.nq > 0:
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)

    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
