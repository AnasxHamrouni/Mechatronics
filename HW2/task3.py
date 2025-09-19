import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import mujoco

dt = 0.001
T = 5.0
n_steps = int(T / dt)
times = np.linspace(0.0, T, n_steps + 1)

force_magnitude = 20.0  # N
force_start_time = 0.5  # s
force_duration = 0.2    # s

model = mujoco.MjModel.from_xml_path("CartPole/task3.xml")
data = mujoco.MjData(model)

slider_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
theta_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "theta")

cart_vel_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_vel")

x_cart_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "x_cart")
v_cart_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "v_cart")
theta_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "theta")
omega_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "omega_theta")
force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "cart_force_sensed")

cart_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cart")
pendulum_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")

data.qpos[0] = 0.0  # cart position
data.qpos[1] = 10.0 * np.pi/180.0  # pendulum angle (converted to radians)
data.qvel[0] = 0.0  # cart velocity
data.qvel[1] = 0.0  # pendulum angular velocity

# Forward the initial state
mujoco.mj_forward(model, data)

# Extract system parameters from the model
cart_mass = model.body_mass[cart_body_id]
pendulum_mass = model.body_mass[pendulum_body_id]
pendulum_length = 0.25  # from the capsule geometry
pendulum_com = 0.125  # from the inertial pos

# Get DOF addresses for joints
slider_dof_adr = model.jnt_dofadr[slider_joint_id]
theta_dof_adr = model.jnt_dofadr[theta_joint_id]

cart_damping = model.dof_damping[slider_dof_adr]
pendulum_damping = model.dof_damping[theta_dof_adr]

cart_frictionloss = 1.0  

gravity = model.opt.gravity[2]

print(f"System parameters:")
print(f"Cart mass: {cart_mass:.3f} kg")
print(f"Pendulum mass: {pendulum_mass:.3f} kg")
print(f"Pendulum length: {pendulum_length:.3f} m")
print(f"Pendulum COM: {pendulum_com:.3f} m")
print(f"Cart damping: {cart_damping:.3f} N·s/m")
print(f"Cart friction loss: {cart_frictionloss:.3f} N")
print(f"Pendulum damping: {pendulum_damping:.3f} N·m·s/rad")
print(f"Gravity: {gravity:.3f} m/s²")

# MuJoCo simulation
def simulate_mujoco():
    # Reset to initial state
    data.qpos[0] = 0.0
    data.qpos[1] = 10.0 * np.pi/180.0
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    data.ctrl[0] = 0.0
    
    mujoco.mj_forward(model, data)
    
    # Initialize arrays to store results
    mujoco_time = np.zeros(n_steps + 1)
    mujoco_x = np.zeros(n_steps + 1)
    mujoco_v = np.zeros(n_steps + 1)
    mujoco_theta = np.zeros(n_steps + 1)
    mujoco_omega = np.zeros(n_steps + 1)
    mujoco_force = np.zeros(n_steps + 1)
    
    # Store initial state
    mujoco_time[0] = data.time
    mujoco_x[0] = data.sensordata[x_cart_sensor_id]
    mujoco_v[0] = data.sensordata[v_cart_sensor_id]
    mujoco_theta[0] = data.sensordata[theta_sensor_id]
    mujoco_omega[0] = data.sensordata[omega_sensor_id]
    mujoco_force[0] = data.sensordata[force_sensor_id]
    
    # Simulation loop
    for i in range(1, n_steps + 1):
        # Apply force for a short duration
        current_time = data.time
        if force_start_time <= current_time < force_start_time + force_duration:
            # Apply a force by setting a velocity control signal
            data.ctrl[cart_vel_actuator_id] = 1.0  
        else:
            data.ctrl[cart_vel_actuator_id] = 0.0
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Store results
        mujoco_time[i] = data.time
        mujoco_x[i] = data.sensordata[x_cart_sensor_id]
        mujoco_v[i] = data.sensordata[v_cart_sensor_id]
        mujoco_theta[i] = data.sensordata[theta_sensor_id]
        mujoco_omega[i] = data.sensordata[omega_sensor_id]
        mujoco_force[i] = data.sensordata[force_sensor_id]
    
    return mujoco_time, mujoco_x, mujoco_v, mujoco_theta, mujoco_omega, mujoco_force

def cart_pendulum_ode(t, y, M, m, L, b_cart, b_pend, g, force_func, friction_loss):
    x, x_dot, theta, theta_dot = y
    
    F = force_func(t)
    # Coulomb friction (static + dynamic)
    if abs(x_dot) < 1e-3:  # Static friction
        friction_force = -min(abs(F), friction_loss) * np.sign(F) if abs(F) > 0 else 0
    else:  # Dynamic friction
        friction_force = -friction_loss * np.sign(x_dot)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # System matrices (from Lagrangian mechanics)
    M_matrix = np.array([
        [M + m, m * L * cos_theta],
        [m * L * cos_theta, m * L**2]
    ])
    
    C_vector = np.array([
        F + friction_force - b_cart * x_dot - m * L * theta_dot**2 * sin_theta,
        m * g * L * sin_theta - b_pend * theta_dot
    ])
    # Solve for accelerations
    try:
        accelerations = np.linalg.solve(M_matrix, C_vector)
    except np.linalg.LinAlgError:
        # Handle singular matrix case
        accelerations = [0, 0]
    
    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]
    
    return [x_dot, x_ddot, theta_dot, theta_ddot]

def force_function(t):
    if force_start_time <= t < force_start_time + force_duration:
        return force_magnitude
    else:
        return 0.0

def simulate_ode():
    y0 = [0.0, 0.0, 10.0 * np.pi/180.0, 0.0]
    
    sol = solve_ivp(
        cart_pendulum_ode, 
        [0, T], 
        y0, 
        args=(cart_mass, pendulum_mass, pendulum_com, cart_damping, pendulum_damping, -gravity, force_function, cart_frictionloss),
        t_eval=times,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

# Run simulations
print("Running MuJoCo simulation")
mujoco_time, mujoco_x, mujoco_v, mujoco_theta, mujoco_omega, mujoco_force = simulate_mujoco()

print("Running ODE simulation")
ode_time, ode_x, ode_v, ode_theta, ode_omega = simulate_ode()

# Convert pendulum angle to degrees 
mujoco_theta_deg = mujoco_theta * 180.0 / np.pi
ode_theta_deg = ode_theta * 180.0 / np.pi

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Position
axes[0, 0].plot(mujoco_time, mujoco_x, label='MuJoCo')
axes[0, 0].plot(ode_time, ode_x, '--', label='ODE')
axes[0, 0].set_ylabel('Cart Position (m)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Velocity
axes[0, 1].plot(mujoco_time, mujoco_v, label='MuJoCo')
axes[0, 1].plot(ode_time, ode_v, '--', label='ODE')
axes[0, 1].set_ylabel('Cart Velocity (m/s)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Angle
axes[1, 0].plot(mujoco_time, mujoco_theta_deg, label='MuJoCo')
axes[1, 0].plot(ode_time, ode_theta_deg, '--', label='ODE')
axes[1, 0].set_ylabel('Pendulum Angle (deg)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Angular velocity
axes[1, 1].plot(mujoco_time, mujoco_omega, label='MuJoCo')
axes[1, 1].plot(ode_time, ode_omega, '--', label='ODE')
axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Applied force
axes[2, 0].plot(mujoco_time, mujoco_force, label='MuJoCo')
axes[2, 0].axvspan(force_start_time, force_start_time + force_duration, alpha=0.3, color='red', label='Force applied')
axes[2, 0].set_ylabel('Applied Force (N)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].legend()
axes[2, 0].grid(True)

# Error between simulations
x_error = np.abs(mujoco_x - ode_x[:len(mujoco_x)])
theta_error = np.abs(mujoco_theta - ode_theta[:len(mujoco_theta)])

axes[2, 1].plot(mujoco_time, x_error, label='Position error')
axes[2, 1].plot(mujoco_time, theta_error, label='Angle error')
axes[2, 1].set_ylabel('Absolute Error')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].legend()
axes[2, 1].grid(True)
axes[2, 1].set_yscale('log')

plt.tight_layout()
plt.suptitle('Cart-Pendulum System: MuJoCo vs ODE Simulation', fontsize=16, y=1.02)
plt.savefig('cart_pendulum_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print maximum errors
print(f"Maximum position error: {np.max(x_error):.6f} m")
print(f"Maximum angle error: {np.max(theta_error):.6f} rad")