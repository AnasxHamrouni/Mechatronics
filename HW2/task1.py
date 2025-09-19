import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import mujoco

xml_file = "MassSpringDamper/task1.xml"

m = 1.0
c = 5.0
k = 200.0
rest_length = 0.20

x0_desired = 0.25
v0_desired = 0.0

# sim params
dt = 0.001
T = 2.0
n_steps = int(T / dt)
times = np.linspace(0.0, T, n_steps + 1)


def load_model_and_ids(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mass_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mass_site")
    wall_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wall_site")
    slider_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    qpos_addr = model.jnt_qposadr[slider_jid]
    qvel_addr = qpos_addr
    return model, data, mass_site_id, wall_site_id, slider_jid, qpos_addr, qvel_addr


def init_mujoco_state_for_mass_site(model, data, mass_site_id, qpos_addr, qvel_addr, x0, v0):
    # zero qpos/qvel and forward to obtain baseline site positions
    data.qpos[:] = 0.0
    # set qvel array to zeros first (length is nvel)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # compute delta in slider coordinate needed to make mass_site world x == x0
    cur_mass_site_x = float(data.site_xpos[mass_site_id, 0])
    delta = x0 - cur_mass_site_x
    data.qpos[qpos_addr] += float(delta)
    data.qvel[qvel_addr] = float(v0)
    mujoco.mj_forward(model, data)
    # return actual values
    mass_x = float(data.site_xpos[mass_site_id, 0])
    mass_v = float(data.qvel[qvel_addr])
    return mass_x, mass_v


def simulate_mujoco(model, data, mass_site_id, wall_site_id, qpos_addr, qvel_addr, x0, v0):
    # initialize so mass_site world x equals desired x0
    mass_x0, mass_v0 = init_mujoco_state_for_mass_site(model, data, mass_site_id, qpos_addr, qvel_addr, x0, v0)
    # arrays (aligned with 'times')
    pos = np.zeros_like(times)
    vel = np.zeros_like(times)
    tarr = np.zeros_like(times)

    # initial sample (t = 0)
    pos[0] = float(data.site_xpos[mass_site_id, 0])
    vel[0] = float(data.qvel[qvel_addr])
    tarr[0] = float(data.time)
    start = time.time()
    for i in range(1, len(times)):
        mujoco.mj_step(model, data)
        pos[i] = float(data.site_xpos[mass_site_id, 0])
        vel[i] = float(data.qvel[qvel_addr])
        tarr[i] = float(data.time)
    wall_x = float(data.site_xpos[wall_site_id, 0])
    return tarr, pos, vel, wall_x


# ODE: mass-spring-damper 
def mass_spring_damper(y, t, m, c, k):
    x, v = y
    dxdt = v
    dvdt = (-k * x - c * v) / m   # equilibrium at x=0
    return [dxdt, dvdt]


def simulate_ode(times, x0, v0, wall_pos):
    y0 = [float(x0), float(v0)]
    sol = odeint(mass_spring_damper, y0, times, args=(m, c, k))
    return sol[:, 0], sol[:, 1]


model, data, mass_site_id, wall_site_id, slider_jid, qpos_addr, qvel_addr = load_model_and_ids(xml_file)

# run MuJoCo sim 
muj_t, muj_x, muj_v, wall_x = simulate_mujoco(model, data, mass_site_id, wall_site_id, qpos_addr, qvel_addr, x0_desired, v0_desired)


# use MuJoCo initial state for ODE so both start from the same point
x0_ode = muj_x[0]
v0_ode = muj_v[0]

ode_x, ode_v = simulate_ode(times, x0_ode, v0_ode, wall_x)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(times, muj_x, label='MuJoCo (mass_site x)', linewidth=2)
ax1.plot(times, ode_x, '--', label='ODE (odeint)', linewidth=2)
ax1.set_ylabel('Position (m)')
ax1.set_title('Mass-Spring-Damper System: Position vs Time')
ax1.legend()
ax1.grid(True)
ax2.plot(times, muj_v, label='MuJoCo (v)', linewidth=2)
ax2.plot(times, ode_v, '--', label='ODE (v)', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Mass-Spring-Damper System: Velocity vs Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300)
plt.show()

# differences
pos_diff = np.abs(muj_x - ode_x)
vel_diff = np.abs(muj_v - ode_v)
print(f"Max position diff: {pos_diff.max():.6e}, Max velocity diff: {vel_diff.max():.6e}")

