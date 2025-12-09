import numpy as np
import mujoco
import matplotlib.pyplot as plt
import csv

def load_motor_specs(csv_path):
    specs = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joint = row["Joint"]
            max_torque = float(row["MaxTorqueNm"])
            max_speed  = float(row["MaxSpeedRPM"])
            gear_ratio = float(row["GearRatio"])
            specs[joint] = dict(
                max_torque=max_torque,
                max_speed=max_speed,
                gear_ratio=gear_ratio,
            )
    return specs

def joint_speed_rpm(model, dq_traj):
    # dq_traj: (N, dof) joint velocities [rad/s]
    # RPM = rad/s * 60 / (2*pi)
    return dq_traj * 60.0 / (2.0 * np.pi)

def plot_motor_envelopes(model, specs, dq_js, tau_js, dq_cs, tau_cs, joint_names, t_list):
    rpm_js = joint_speed_rpm(model, dq_js)
    rpm_cs = joint_speed_rpm(model, dq_cs)

    dof = len(joint_names)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6)) 
    axes = axes.flatten()

    for j, jname in enumerate(joint_names):
        ax = axes[j]
        if jname not in specs:
            ax.set_visible(False)
            continue

        s = specs[jname]
        stall = s["max_torque"]
        max_rpm = s["max_speed"]
        rpm_line = np.linspace(0, max_rpm, 100)
        torque_line = stall * (1.0 - rpm_line / max_rpm)

        ax.plot(rpm_line, torque_line, "k-", label="Stall–no-load line")
        ax.scatter(rpm_js[:, j], np.abs(tau_js[:, j]), s=4, c="b", alpha=0.4, label="JS traj" if j == 0 else None)
        ax.scatter(rpm_cs[:, j], np.abs(tau_cs[:, j]), s=4, c="g", alpha=0.4, label="CS traj" if j == 0 else None)
        ax.set_title(jname)
        ax.set_xlabel("Speed [rpm]")
        ax.set_ylabel("Torque [Nm]")
        ax.set_xlim(0, max_rpm * 1.1)
        ax.set_ylim(0, stall * 1.2)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()

def ik_position_only(model, body_name, goal_pos, q_seed,
                     max_iters=80, tol=1e-4, step_scale=0.5, damping=1e-4):
    data = mujoco.MjData(model)
    ndof = model.nq
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    q = q_seed.copy()
    for _ in range(max_iters):
        data.qpos[:ndof] = q
        mujoco.mj_forward(model, data)

        p = data.xpos[body_id].copy()
        err = goal_pos - p
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = jacp[:, :ndof]        
        JJt = J @ J.T
        dq_task = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)
        q += step_scale * dq_task
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                adr = model.jnt_qposadr[j]
                low, high = model.jnt_range[j]
                q[adr] = np.clip(q[adr], low, high)

    return q

XML_PATH = "ur_description/urdf/mjmodel.xml"  
DT       = 0.002
T_TOTAL  = 1                             
N        = int(T_TOTAL / DT)

def get_joint_slice(model):
    return slice(0, 6)

def min_jerk_positions(q0, q1, T, N):
    dof = q0.size
    q_traj = np.zeros((N, dof))
    t_list = np.linspace(0.0, T, N)
    for i, t in enumerate(t_list):
        tau = np.clip(t / T, 0.0, 1.0)
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        q_traj[i] = q0 + s * (q1 - q0)
    return q_traj, t_list

def diff_traj(q_traj, dt):
    dq = np.zeros_like(q_traj)
    ddq = np.zeros_like(q_traj)
    dq[1:-1]  = (q_traj[2:] - q_traj[:-2]) / (2 * dt)
    dq[0]     = (q_traj[1]  - q_traj[0]) / dt
    dq[-1]    = (q_traj[-1] - q_traj[-2]) / dt

    ddq[1:-1] = (dq[2:] - dq[:-2]) / (2 * dt)
    ddq[0]    = (dq[1]  - dq[0]) / dt
    ddq[-1]   = (dq[-1] - dq[-2]) / dt
    return dq, ddq

# Cartesian trajectory via IK 
def build_cartesian_traj(model, q_start, q_end, T, N):
    dof = 6
    q_traj   = np.zeros((N, dof))
    data     = mujoco.MjData(model)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool0")

    # start pose
    data.qpos[:dof] = q_start
    mujoco.mj_forward(model, data)
    T_start = np.eye(4)
    T_start[:3, :3] = data.xmat[tcp_id].reshape(3, 3)
    T_start[:3, 3]  = data.xpos[tcp_id]

    # end pose
    data.qpos[:dof] = q_end
    mujoco.mj_forward(model, data)
    T_end = np.eye(4)
    T_end[:3, :3] = data.xmat[tcp_id].reshape(3, 3)
    T_end[:3, 3]  = data.xpos[tcp_id]

    for k in range(N):
        t   = k * T / (N - 1)
        tau = np.clip(t / T, 0.0, 1.0)
        p = (1 - tau) * T_start[:3, 3] + tau * T_end[:3, 3]
        if k == 0:
            q_seed = q_start
        else:
            q_seed = q_traj[k-1]
        q = ik_position_only(model, "tool0", p, q_seed)
        q_traj[k] = q

    dq_traj, ddq_traj = diff_traj(q_traj, DT)

    print("CS traj stats:")
    print("  q   max abs:", np.max(np.abs(q_traj)))
    print("  dq  max abs:", np.max(np.abs(dq_traj)))
    print("  ddq max abs:", np.max(np.abs(ddq_traj)))
    return q_traj, dq_traj, ddq_traj

def compute_inverse_dynamics(model, q_traj, dq_traj, ddq_traj, label=""):
    data = mujoco.MjData(model)
    dof  = q_traj.shape[1]
    tau  = np.zeros_like(q_traj)
    dq_safe  = np.clip(dq_traj,  -2.0,  2.0)   
    ddq_safe = np.clip(ddq_traj, -5.0,  5.0)    

    for k in range(q_traj.shape[0]):
        data.qpos[:dof] = q_traj[k]
        data.qvel[:dof] = dq_safe[k]
        data.qacc[:dof] = ddq_safe[k]
        mujoco.mj_inverse(model, data)
        tau[k] = data.qfrc_inverse[:dof]

    abs_tau  = np.abs(tau)
    mean_tau = abs_tau.mean(axis=0)
    max_tau  = abs_tau.max(axis=0)
    print(f"\n{label} torque stats:")
    for j in range(dof):
        print(f"  joint {j}: mean |tau| = {mean_tau[j]:.4e}, max |tau| = {max_tau[j]:.4e}")

    return tau

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    j_slice = get_joint_slice(model)
    dof = j_slice.stop - j_slice.start

    # JS trajectory
    q0 = np.array([0, -np.pi/4, np.pi/4, -np.pi/3, np.pi/4, 0.0])
    q1 = np.array([np.pi/6, -np.pi/3, np.pi/3, -np.pi/3, np.pi/3, np.pi/6])
    q_js, t_list = min_jerk_positions(q0, q1, T_TOTAL, N)
    dq_js, ddq_js = diff_traj(q_js, DT)
    print("JS traj stats:")
    print("  q   max abs:", np.max(np.abs(q_js)))
    print("  dq  max abs:", np.max(np.abs(dq_js)))
    print("  ddq max abs:", np.max(np.abs(ddq_js)))
    tau_js = compute_inverse_dynamics(model, q_js, dq_js, ddq_js, label="JS")

    # CS trajectory
    q_cs, dq_cs, ddq_cs = build_cartesian_traj(model, q0, q1, T_TOTAL, N)
    tau_cs = compute_inverse_dynamics(model, q_cs, dq_cs, ddq_cs, label="CS")
    specs = load_motor_specs("motors-data.csv")
    joint_names = [model.joint(i).name for i in range(dof)]
    plot_motor_envelopes(model, specs,
                        dq_js, tau_js,
                        dq_cs, tau_cs,
                        joint_names, t_list)


    # Plot
    joint_names = [model.joint(i).name for i in range(dof)]
    time = t_list
    fig, axes = plt.subplots(dof, 1, figsize=(8, 2*dof), sharex=True)
    for j in range(dof):
        ax = axes[j]
        ax.plot(time, tau_js[:, j], label="JS torque")
        ax.plot(time, tau_cs[:, j], label="CS torque", linestyle="--")
        ax.set_ylabel(f"{joint_names[j]} τ [Nm]")
        ax.grid(True)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()
