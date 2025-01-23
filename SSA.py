#!/usr/bin/env python
# coding: utf-8

# In[1]:


#cauchy

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
from random import sample

# Load your data
file_path = '80_dataset_with_fixed_cycles.csv'
data = pd.read_csv(file_path)
data = data[data['Peptide Ratio'] == '80_background_edit']

# Extract necessary data
efp_in = data['EFP_in'].values
ee = data['Endosomal Escape'].values
inc_efp_in = np.diff(efp_in) / 5
inc_ee = np.diff(ee) / 5
inc_efp_out = inc_efp_in - inc_ee
n_steps = len(inc_efp_in)

# Define the Bayesian hierarchical model
with pm.Model() as hierarchical_model:
    # Global hyperpriors
    global_mu = pm.Cauchy("global_mu", alpha=0.3255936 , beta=0.218884)
    global_sigma = pm.Cauchy("global_sigma", alpha=0.2383414 , beta=0.1373218 )
    #global_dexit = pm.Cauchy("global_dexit", alpha=0.03666491 , beta=0.1004623)
    global_dexit =  pm.Normal("global_dexit", mu=0.03666491, sigma=0.1)

    # Local priors for each time step
    mu = pm.Normal("mu", mu=global_mu, sigma=0.1, shape=n_steps)
    sigma = pm.Normal("sigma", mu=global_sigma, sigma=0.1, shape=n_steps)
    dexit = pm.Normal("dexit", mu=global_dexit, sigma=0.1, shape=n_steps)

    # Likelihoods for the observed increments
    likelihood_efp_in = pm.Normal("likelihood_efp_in", mu=mu, sigma=0.3, observed=inc_efp_in)
    likelihood_ee = pm.Normal("likelihood_ee", mu=sigma, sigma=0.3, observed=inc_ee)
    likelihood_efp_out = pm.Normal("likelihood_efp_out", mu=dexit, sigma=0.2, observed=inc_efp_out)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, target_accept=0.9, chains=5)

# Extract posterior means for each time step
mu_estimates = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
sigma_estimates = trace.posterior["sigma"].mean(dim=["chain", "draw"]).values
dexit_estimates = trace.posterior["dexit"].mean(dim=["chain", "draw"]).values


# In[2]:


plt.rcParams['figure.constrained_layout.use'] = True
az.plot_trace(trace)
az.summary(trace)


# In[3]:


az.plot_trace(trace, var_names=["mu", "sigma","dexit"])


# In[4]:


tf = 4
tstep = n_steps
beta = 0.000
peptide_ratio = 80  # Set the peptide ratio directly
ratio_exp = 6.892
fixed_siRNA_amount = 120

exp_value = ratio_exp
initial_FP = exp_value * fixed_siRNA_amount

# initial_conditions = np.array([
#                    initial_FP,
#                     fixed_siRNA_amount,
#                     efp_in[0] *fixed_siRNA_amount/100 ,
#                     ee[0] *fixed_siRNA_amount/100 ,
#                   (efp_in[0] - ee[0])*fixed_siRNA_amount/100
# ] , dtype=float)
initial_conditions = np.array([
                  initial_FP-fixed_siRNA_amount,
                    0,
                    #100-(efp_in[0] *fixed_siRNA_amount/100 ),
                    100-efp_in[0] ,
                    ee[0] *fixed_siRNA_amount/100 ,
                  (efp_in[0] - ee[0])*fixed_siRNA_amount/100
] , dtype=float)

print(initial_conditions)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gillespie_extended_with_posteriors(
    mu_estimates, sigma_estimates, dexit_estimates,
    initial_state, t_max, t_points
):
    A, B, C, D, E = initial_state

    times = [0]
    states = [[A, B, C, D, E]]

    while times[-1] < t_max:
        # Find the closest time step index for posterior parameters
        t_idx = np.searchsorted(t_points, times[-1], side="right") - 1

        # Use the posterior means for the current time step
        mu = mu_estimates[t_idx]
        sigma = sigma_estimates[t_idx]
        dexit = dexit_estimates[t_idx]
        # Define a correction factor for the early time steps (e.g., first 8–9 steps)
        # correction_factor = 0.7  # Adjust as necessary

        # # Apply the correction for the first 8–9 time steps
        # if t_idx < 9:  # First 8–9 steps
        #     mu = mu_estimates[t_idx] * correction_factor
        #     sigma = sigma_estimates[t_idx] * correction_factor
        #     dexit = dexit_estimates[t_idx] * correction_factor
        # else:
        #     mu = mu_estimates[t_idx]
        #     sigma = sigma_estimates[t_idx]
        #     dexit = dexit_estimates[t_idx]


        # Reaction rates
        rates = [
            mu * A * B,  # A + B -> C
            0 * B,       # B -> D (assume no transition here as per original code)
            sigma * C,   # C -> D
            dexit * C    # C -> E
        ]
        total_rate = sum(rates)

        if total_rate <= 0:
            break

        # Time to next reaction
        tau = np.random.exponential(1 / total_rate)
        times.append(times[-1] + tau)

        # Determine which reaction occurs
        reaction_probabilities = np.cumsum(rates) / total_rate
        reaction_choice = np.searchsorted(reaction_probabilities, np.random.rand())

        # Stoichiometry: change in concentrations
        if reaction_choice == 0:  # A + B -> C
            A -= 1
            B -= 1
            C += 1
        elif reaction_choice == 1:  # B -> D
            pass
        elif reaction_choice == 2:  # C -> D
            C -= 1
            D += 1
        elif reaction_choice == 3:  # C -> E
            C -= 1
            E += 1

        # Enforce non-negativity constraints
        C = max(C, 0)
        D = max(D, 0)
        E = max(E, 0)

        # Enforce upper boundary (max value = 100)
        C = min(C, 100)
        D = min(D, 100)
        E = min(E, 100)

        states.append([A, B, C, D, E])

    return np.array(times), np.array(states)


# Summary statistics: Extract concentration of C, D, E
def summary_statistics(times, states, t_points, species_idx):
    species_concentration = states[:, species_idx]  # Index for C, D, or E
    return np.interp(t_points, times, species_concentration)

# Normalize data: Z-score normalization
def normalize(data):
    std = np.std(data)
    if std == 0:
        return data - np.mean(data)  # Normalize to zero mean only
    return (data - np.mean(data)) / std

def abc_extended_all_species_with_posteriors(
    observed_data, t_points, mu_estimates, sigma_estimates, dexit_estimates,
    num_simulations, tolerance, t_max, initial_state
):

    # Unpack observed data (C, D, E)
    observed_data_C, observed_data_D, observed_data_E = observed_data

    # Initialize storage for accepted parameters and simulated curves
    accepted_params = []
    accepted_simulated_C = []
    accepted_simulated_D = []
    accepted_simulated_E = []

    for simulation in range(num_simulations):

        times, states = gillespie_extended_with_posteriors(
            mu_estimates, sigma_estimates, dexit_estimates,
            initial_state=initial_state, t_max=t_max, t_points=t_points
        )

        # Compute summary statistics for each species (C, D, E)
        simulated_C = summary_statistics(times, states, t_points, species_idx=2)  # C concentrations
        simulated_D = summary_statistics(times, states, t_points, species_idx=3)  # D concentrations
        simulated_E = summary_statistics(times, states, t_points, species_idx=4)  # E concentrations

        # Normalize the observed data for D
        observed_data_D = observed_data[1]  # Extract observed data for D
        observed_data_D_interp = normalize(np.interp(t_points, np.linspace(0, t_max, len(observed_data_D)), observed_data_D))

        # Normalize the simulated data for D
        simulated_D_normalized = normalize(simulated_D)

        # Calculate distance for D
        distance_D = np.linalg.norm(simulated_D_normalized - observed_data_D_interp)

        #print(f"Simulation {simulation+1}/{num_simulations}: Distance for D = {distance_D}")

        # Accept parameters and save paths if distance for D is smaller than tolerance
        if distance_D < tolerance:
            #accepted_params.append(sampled_params)
            accepted_simulated_C.append(simulated_C)
            accepted_simulated_D.append(simulated_D)
            accepted_simulated_E.append(simulated_E)

    # Return accepted parameters and accepted simulated curves
    return (
        #np.array(accepted_params),
        np.array(accepted_simulated_C),
        np.array(accepted_simulated_D),
        np.array(accepted_simulated_E)
    )


# Load the observed data from CSV
def load_observed_data(csv_file):
    data = pd.read_csv(csv_file)
    data=data[data['Peptide Ratio'] == '80_background_edit']
    efp_in = data['EFP_in'].values
    ee = data['Endosomal Escape'].values
    efp_out = efp_in - ee
    C = efp_in  # C is the EFP_in
    D = ee      # D is the Endosomal Escape
    E = efp_out # E is the difference (efp_in - Endosomal Escape)

    # Return as tuple for C, D, E
    return C, D, E

# Main setup
if __name__ == "__main__":
    # Load the observed data from a CSV file
    csv_file = '80_dataset_with_fixed_cycles.csv'  # Replace with the path to your CSV file
    observed_data = load_observed_data(csv_file)  # Returns C, D, E as tuple

    # Set time points for interpolation
    t_points = np.linspace(0, 4, n_steps)

    # Number of simulations and tolerance
    num_simulations = 6000
    tolerance = 1 # Adjust as necessary

    # Initial conditions
    initial_conditions = initial_conditions

    # Set the maximum time for simulation
    t_max = 4  # Max time for simulation (based on the data)

    # Perform ABC using data for C, D, and E
    all_simulated_C, all_simulated_D, all_simulated_E = abc_extended_all_species_with_posteriors(
        observed_data, t_points, mu_estimates, sigma_estimates, dexit_estimates,
        num_simulations, tolerance, t_max, initial_conditions
    )
    # Interpolate the observed data to match t_points (48 points)
    observed_data_C_interp = np.interp(t_points, np.linspace(0, 4, len(observed_data[0])), observed_data[0])
    observed_data_D_interp = np.interp(t_points, np.linspace(0, 4, len(observed_data[1])), observed_data[1])
    observed_data_E_interp = np.interp(t_points, np.linspace(0, 4, len(observed_data[2])), observed_data[2])





# In[6]:


# Plot the observed and simulated curves
plt.figure(figsize=(10,8))

# Plot for C (EFP_in)
plt.subplot(3, 1, 1)
for sim in all_simulated_C:
    plt.plot(t_points, 100-sim, color='lightblue', alpha=0.1)
plt.plot(t_points, observed_data_C_interp, label="Observed EFP_in", marker='o', color='blue')
plt.xlabel('Time (Hours)')
plt.ylabel('EFP_in')
plt.title('Simulated vs Observed EFP_in')
plt.legend()
plt.ylim(0, 105)

# Plot for D (Endosomal Escape)
plt.subplot(3, 1, 2)
for sim in all_simulated_D:
    plt.plot(t_points, sim, color='gray', alpha=0.1)
plt.plot(t_points, observed_data_D_interp, label="Observed Endosomal Escape", marker='o', color='black')
plt.xlabel('Time (Hours)')
plt.ylabel('Endosomal Escape %')
plt.title('Simulated vs Observed Endosomal Escape Percentage')
plt.legend()
plt.ylim(0, 105)

# Plot for E (efp_out)
plt.subplot(3, 1, 3)

for sim in all_simulated_E:
    plt.plot(t_points, sim, color='lightcoral', alpha=0.1)
plt.plot(t_points, observed_data_E_interp, label="Observed EFP_out", marker='o', color='red')
plt.xlabel('Time (Hours)')
plt.ylabel('EFP_out')
plt.title('Simulated vs Observed EFP_out')
plt.legend()
plt.ylim(0, 105)

plt.tight_layout()
plt.show()


# In[7]:


# Plot the observed and simulated curves on one plot
plt.figure(figsize=(10, 6))

# Plot all simulations and observed data on the same plot

# EFP_in
for sim in all_simulated_C:
    plt.plot(t_points, 100-sim, color='lightblue', alpha=0.1)
plt.plot(t_points, observed_data_C_interp, label="Observed EFP_in", marker='o', color='blue')

# Endosomal Escape Percentage
for sim in all_simulated_D:
    plt.plot(t_points, sim, color='gray', alpha=0.1)
plt.plot(t_points, observed_data_D_interp, label="Observed Endosomal Escape", marker='o', color='black')

# EFP_out
for sim in all_simulated_E:
    plt.plot(t_points, sim, color='lightcoral', alpha=0.1)
plt.plot(t_points, observed_data_E_interp, label="Observed EFP_out", marker='o', color='red')

# Labels and title
plt.xlabel('Time (Hours)')
plt.ylabel('Percentage')
plt.title('Simulated vs Observed Data (EFP_in, Endosomal Escape, EFP_out)')
plt.legend()

# Set the y-axis to 0-105 for all data
plt.ylim(0, 105)

plt.tight_layout()
plt.show()


# In[ ]:




