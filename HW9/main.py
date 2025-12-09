import os
import mujoco
import mujoco.viewer
from lxml import etree


def set_joint_armatures(xml_path):
    # Load XML file and build XML tree
    tree = etree.parse(xml_path)
    root = tree.getroot()
    joints = root.xpath("//joint")
    armature_values = {
        "shoulder_pan_joint": 0.6059,
        "shoulder_lift_joint": 0.6059,
        "elbow_joint": 0.6457,
        "wrist_1_joint": 0.1265,
        "wrist_2_joint": 0.1265,
        "wrist_3_joint": 0.1265,
    }

    print(f"Found {len(joints)} joints:")
    for i, joint in enumerate(joints, 1):
        joint_name = joint.get("name", "unnamed")
        armature_value = armature_values.get(joint_name, 0.05)
        # Set/overwrite armature
        joint.set("armature", str(armature_value))
        print(f"  Joint {i}: {joint_name} -> armature={armature_value}")

    # Save modified XML
    tree.write(xml_path, pretty_print=True, xml_declaration=True, encoding="utf-8")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "ur_description/urdf/mjmodel.xml")
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    set_joint_armatures(xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("Model loaded.")
    print(f"Number of joints: {model.njnt}")
    print("Compiled joint armatures:")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"  {j}: {name} -> armature={model.dof_armature[j]}")

    if model.nq > 0:
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
