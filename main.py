import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system parameters
J = 10  # Moment of inertia (kg*m^2)
D = 0.5  # Damping coefficient (N*m*s)
gamma = 1.4  # Specific heat ratio
R = 287  # Specific gas constant (J/(kg*K))
rho = 1.225  # Air density at sea level (kg/m^3)
V = 50  # Fluid velocity (m/s)
R_turbine = 1  # Turbine radius (m)
A = np.pi * R_turbine ** 2  # Effective surface area (m^2)

# Example parameters
examples = [
    {"mass_flow_rate": 1, "T_in": 500, "P_in": 2e5, "alpha": np.pi / 6},
    {"mass_flow_rate": 1.2, "T_in": 520, "P_in": 2.2e5, "alpha": np.pi / 4},
    {"mass_flow_rate": 1.5, "T_in": 550, "P_in": 2.5e5, "alpha": np.pi / 3},
    {"mass_flow_rate": 1.8, "T_in": 580, "P_in": 2.8e5, "alpha": np.pi / 6},
    {"mass_flow_rate": 2, "T_in": 600, "P_in": 3e5, "alpha": np.pi / 4}
]


def turbine_dynamics(t, y, mass_flow_rate, T_in, P_in, alpha):
    omega = y[0]
    T_out = T_in * (1 - (D * omega) / (mass_flow_rate * R * T_in / (gamma - 1)))
    h_out = R * T_out / (gamma - 1)
    P_out = P_in * (T_out / T_in) ** (gamma / (gamma - 1))
    h_in = R * T_in / (gamma - 1)
    torque = rho * A * V ** 2 * R_turbine * np.sin(alpha)
    d_omega_dt = (torque - D * omega) / J
    return [d_omega_dt, P_out, T_out, h_out]


# Set initial conditions and simulation time
y0 = [0, 2e5, 500, R * 500 / (
            gamma - 1)]  # Initial angular velocity (rad/s), initial pressure (Pa), initial temperature (K), initial enthalpy (J/kg)
t_span = (0, 20)  # Simulation time span (seconds)
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Evaluation times for results

# Solve the differential equations for each example
results = []
for params in examples:
    sol = solve_ivp(turbine_dynamics, t_span, y0, t_eval=t_eval,
                    args=(params['mass_flow_rate'], params['T_in'], params['P_in'], params['alpha']), vectorized=True)
    results.append(sol)

# Plot the results
for i, sol in enumerate(results):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Example {i + 1}')
    titles = ['Angular Velocity (rad/s)', 'Output Pressure (Pa)', 'Output Temperature (K)', 'Output Enthalpy (J/kg)']

    ax1 = axs[0]
    ax1.plot(sol.t, sol.y[0], label='Angular Velocity (rad/s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title(titles[0])
    ax1.grid(True)

    ax2 = axs[1]
    ax2.plot(sol.t, sol.y[1], label='Output Pressure (Pa)', color='tab:red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Output Pressure (Pa)')
    ax2.set_title(titles[1])
    ax2.grid(True)

    ax3 = axs[2]
    ax3.plot(sol.t, sol.y[2], label='Output Temperature (K)', color='tab:green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Output Temperature (K)')
    ax3.set_title(titles[2])
    ax3.grid(True)

    ax4 = axs[3]
    ax4.plot(sol.t, sol.y[3], label='Output Enthalpy (J/kg)', color='tab:orange')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Output Enthalpy (J/kg)')
    ax4.set_title(titles[3])
    ax4.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
