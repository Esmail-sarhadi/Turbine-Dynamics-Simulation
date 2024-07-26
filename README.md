
# 🌪️ Turbine Dynamics Simulation

This project simulates the dynamics of a turbine under different operational conditions. The simulation considers parameters such as mass flow rate, inlet temperature, inlet pressure, and flow angle to determine the turbine's behavior over time.

## 📑 Table of Contents 
     
- [📖 Project Overview](#project-overview)
- [⚙️ Installation](#installation)   
- [📚 Usage](#usage)        
- [🔬 Algorithm Explanation](#algorithm-explanation)                
- [🔢 Examples](#examples)                       
- [🤝 Contributing](#contributing)      
- [📄 License](#license)  
- [💖 Donation](#donation)    
 
## 📖 Project Overview

The goal of this project is to model and simulate the dynamic response of a turbine. The system is described by a set of differential equations that account for the angular velocity of the turbine, output pressure, output temperature, and output enthalpy.

## ⚙️ Installation

To run this project, you need Python 3.x installed on your system along with the necessary libraries. You can check your Python version by running:

```bash
python --version
```

### Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install numpy matplotlib scipy
```

### Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/esmail-sarhadi/Turbine-Dynamics-Simulation.git
cd turbine-dynamics-simulation
```

## 📚 Usage

The program simulates the turbine dynamics for different sets of input parameters. To run the simulation, execute the script:

```bash
python turbine_simulation.py
```

The script will simulate the turbine dynamics for five different sets of example parameters and plot the results.

### Example Output

The script produces plots showing the time evolution of:

- Angular Velocity (rad/s)
- Output Pressure (Pa)
- Output Temperature (K)
- Output Enthalpy (J/kg)

Each plot corresponds to a different set of example parameters.

## 🔬 Algorithm Explanation

The simulation uses the following parameters and differential equations to model the turbine dynamics:

- **System Parameters**:
  - Moment of inertia (J)
  - Damping coefficient (D)
  - Specific heat ratio (gamma)
  - Specific gas constant (R)
  - Air density (rho)
  - Fluid velocity (V)
  - Turbine radius (R_turbine)
  - Effective surface area (A)

- **Differential Equations**:
  - The angular velocity (\(\omega\)) is governed by the torque generated by the fluid flow and the damping force.
  - The output temperature (\(T_{out}\)), output pressure (\(P_{out}\)), and output enthalpy (\(h_{out}\)) are derived from the turbine dynamics.

The script uses `solve_ivp` from the `scipy.integrate` module to solve the system of differential equations.

## 🔢 Examples

The script includes five example parameter sets that illustrate the turbine's response under different operational conditions:

1. `mass_flow_rate=1`, `T_in=500K`, `P_in=200kPa`, `alpha=30 degrees`
2. `mass_flow_rate=1.2`, `T_in=520K`, `P_in=220kPa`, `alpha=45 degrees`
3. `mass_flow_rate=1.5`, `T_in=550K`, `P_in=250kPa`, `alpha=60 degrees`
4. `mass_flow_rate=1.8`, `T_in=580K`, `P_in=280kPa`, `alpha=30 degrees`
5. `mass_flow_rate=2`, `T_in=600K`, `P_in=300kPa`, `alpha=45 degrees`

Each example runs the simulation for a time span of 20 seconds, evaluating the results at 500 equally spaced points.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please fork the repository and create a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💖 Donation
If you found this project helpful, consider making a donation:

<a href="https://nowpayments.io/donation?api_key=REWCYVC-A1AMFK3-QNRS663-PKJSBD2&source=lk_donation&medium=referral" target="_blank">
     <img src="https://nowpayments.io/images/embeds/donation-button-black.svg" alt="Crypto donation button by NOWPayments">
</a>
