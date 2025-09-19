import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import mujoco

xml_file = "DoubleMass/task2.xml"  
dt = 0.001
T = 2.0
n_steps = int(T / dt)
times = np.linspace(0.0, T, n_steps + 1)

k_left = 300.0
k_mid = 400.0
k_right = 300.0
c_left = 5.0
c_mid = 5.0
c_right = 5.0

m1 = 1.0
m2 = 1.0

perturb_x1 = 0.02   
perturb_x2 = 0.00 
init_v1 = 0.0
init_v2 = 0.0


def load_model_and_ids(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mass1_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mass1_site")
    mass2_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mass2_site")
    wall_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wall_left_site")
    wall_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wall_right_site")

    jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider1")
    jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider2")
    qpos_addr1 = int(model.jnt_qposadr[jid1])
    qpos_addr2 = int(model.jnt_qposadr[jid2])
    qvel_addr1 = qpos_addr1
    qvel_addr2 = qpos_addr2

    return model, data, mass1_site_id, mass2_site_id, wall_left_id, wall_right_id, jid1, jid2, qpos_addr1, qpos_addr2, qvel_addr1, qvel_addr2


def compute_equilibrium_positions(model, data, mass1_site_id, mass2_site_id, wall_left_id, wall_right_id):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    x1_eq = float(data.site_xpos[mass1_site_id, 0])
    x2_eq = float(data.site_xpos[mass2_site_id, 0])
    wall_left_x = float(data.site_xpos[wall_left_id, 0])
    wall_right_x = float(data.site_xpos[wall_right_id, 0])
    return x1_eq, x2_eq, wall_left_x, wall_right_x


def init_state_perturb(model, data, qpos_addr1, qpos_addr2, qvel_addr1, qvel_addr2, x1_eq, x2_eq,
                       dx1=0.0, dx2=0.0, v1=0.0, v2=0.0):
    # put system at equilibrium first
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # compute the world x positions of sites and the current slider qpos mapping
    # We can apply small offsets in qpos addresses to move sliders relative to that rest.
    data.qpos[qpos_addr1] = (x1_eq - float(data.site_xpos[mass1_site_id, 0])) + dx1 + data.qpos[qpos_addr1]
    data.qpos[qpos_addr2] = (x2_eq - float(data.site_xpos[mass2_site_id, 0])) + dx2 + data.qpos[qpos_addr2]

    data.qvel[qvel_addr1] = float(v1)
    data.qvel[qvel_addr2] = float(v2)
    mujoco.mj_forward(model, data)
    return


def simulate_mujoco(model, data, mass1_site_id, mass2_site_id, qpos_addr1, qpos_addr2, qvel_addr1, qvel_addr2):
    pos1 = np.zeros_like(times)
    pos2 = np.zeros_like(times)
    vel1 = np.zeros_like(times)
    vel2 = np.zeros_like(times)
    tarr = np.zeros_like(times)

    # initial sample (t = 0)
    pos1[0] = float(data.site_xpos[mass1_site_id, 0])
    pos2[0] = float(data.site_xpos[mass2_site_id, 0])
    vel1[0] = float(data.qvel[qvel_addr1])
    vel2[0] = float(data.qvel[qvel_addr2])
    tarr[0] = float(data.time)

    for i in range(1, len(times)):
        mujoco.mj_step(model, data)
        pos1[i] = float(data.site_xpos[mass1_site_id, 0])
        pos2[i] = float(data.site_xpos[mass2_site_id, 0])
        vel1[i] = float(data.qvel[qvel_addr1])
        vel2[i] = float(data.qvel[qvel_addr2])
        tarr[i] = float(data.time)
    return tarr, pos1, pos2, vel1, vel2


def two_mass_ode(y, t, m1, m2, kL, kM, kR, cL, cM, cR):
    u1, v1, u2, v2 = y
    # Forces: left spring on mass1 = -kL * u1 ; mid spring = -kM*(u1 - u2)
    # damping similarly with relative velocities
    f1 = -kL * u1 - kM * (u1 - u2) - cL * v1 - cM * (v1 - v2)
    f2 = -kR * u2 - kM * (u2 - u1) - cR * v2 - cM * (v2 - v1)
    a1 = f1 / m1
    a2 = f2 / m2
    return [v1, a1, v2, a2]


def simulate_ode(times, x1_eq, x2_eq, muj_pos1, muj_pos2, muj_vel1, muj_vel2):
    # initial displacement from equilibrium
    u1_0 = muj_pos1[0] - x1_eq
    u2_0 = muj_pos2[0] - x2_eq
    v1_0 = muj_vel1[0]
    v2_0 = muj_vel2[0]
    y0 = [u1_0, v1_0, u2_0, v2_0]

    sol = odeint(two_mass_ode, y0, times, args=(m1, m2, k_left, k_mid, k_right, c_left, c_mid, c_right))
    # convert back to absolute positions: x = eq + u
    x1 = sol[:, 0] + x1_eq
    v1 = sol[:, 1]
    x2 = sol[:, 2] + x2_eq
    v2 = sol[:, 3]
    return x1, v1, x2, v2


(model, data,
    mass1_site_id, mass2_site_id,
    wall_left_id, wall_right_id,
    jid1, jid2,
    qpos_addr1, qpos_addr2, qvel_addr1, qvel_addr2) = load_model_and_ids(xml_file)

# compute equilibrium positions
x1_eq, x2_eq, wall_left_x, wall_right_x = compute_equilibrium_positions(
    model, data, mass1_site_id, mass2_site_id, wall_left_id, wall_right_id
)
print(f"Equilibrium positions (from MuJoCo rest): x1={x1_eq:.4f}, x2={x2_eq:.4f}, wallL={wall_left_x:.4f}, wallR={wall_right_x:.4f}")

# initialize state with small perturbation (so it will move)
# we add perturb to the slider qpos addresses relative to the computed rest
# here we add dx to mass positions (approx); this is a simple approach
# Alternatively set qvel to give an initial velocity impulse.
data.qpos[:] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# compute how many meters each qpos increment is in world coords by reading site positions,
data.qpos[qpos_addr1] += perturb_x1
data.qpos[qpos_addr2] += perturb_x2
data.qvel[qvel_addr1] = init_v1
data.qvel[qvel_addr2] = init_v2
mujoco.mj_forward(model, data)

# run MuJoCo simulation
muj_t, muj_x1, muj_x2, muj_v1, muj_v2 = simulate_mujoco(
    model, data, mass1_site_id, mass2_site_id, qpos_addr1, qpos_addr2, qvel_addr1, qvel_addr2
)

# run ODE using MuJoCo initial state as baseline (so both start from same initial conditions)
ode_x1, ode_v1, ode_x2, ode_v2 = simulate_ode(times, x1_eq, x2_eq, muj_x1, muj_x2, muj_v1, muj_v2)

# plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axp1 = axs[0, 0]
axp2 = axs[0, 1]
axv1 = axs[1, 0]
axv2 = axs[1, 1]

axp1.plot(times, muj_x1, label='MuJoCo x1', linewidth=2)
axp1.plot(times, ode_x1, '--', label='ODE x1', linewidth=2)
axp1.set_title('Mass 1 Position')
axp1.set_ylabel('x (m)')
axp1.legend()
axp1.grid(True)

axp2.plot(times, muj_x2, label='MuJoCo x2', linewidth=2)
axp2.plot(times, ode_x2, '--', label='ODE x2', linewidth=2)
axp2.set_title('Mass 2 Position')
axp2.legend()
axp2.grid(True)

axv1.plot(times, muj_v1, label='MuJoCo v1', linewidth=2)
axv1.plot(times, ode_v1, '--', label='ODE v1', linewidth=2)
axv1.set_title('Mass 1 Velocity')
axv1.set_ylabel('v (m/s)')
axv1.set_xlabel('Time (s)')
axv1.legend()
axv1.grid(True)

axv2.plot(times, muj_v2, label='MuJoCo v2', linewidth=2)
axv2.plot(times, ode_v2, '--', label='ODE v2', linewidth=2)
axv2.set_title('Mass 2 Velocity')
axv2.set_xlabel('Time (s)')
axv2.legend()
axv2.grid(True)

plt.tight_layout()
plt.savefig('two_masses_comparison.png', dpi=300)
plt.show()

# print max differences
pos_diff1 = np.abs(muj_x1 - ode_x1)
pos_diff2 = np.abs(muj_x2 - ode_x2)
vel_diff1 = np.abs(muj_v1 - ode_v1)
vel_diff2 = np.abs(muj_v2 - ode_v2)

print(f"Mass1 - max pos diff: {pos_diff1.max():.6e}, max vel diff: {vel_diff1.max():.6e}")
print(f"Mass2 - max pos diff: {pos_diff2.max():.6e}, max vel diff: {vel_diff2.max():.6e}")
