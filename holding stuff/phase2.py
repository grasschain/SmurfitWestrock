import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations=10000, random_seed=42):
    """
    Computes the optimal Q1 (initial feed quantity) using a simulation-based approach.

    Parameters:
        final_demand (float): The final required demand (D) for the job.
        machine_sequence (list of tuples): Each tuple is (machine_name, waste_mean, waste_std),
                                           with waste values in percentages.
        Cu (float): Underage cost (cost per unit short).
        Co (float): Overage cost (cost per unit leftover).
        n_simulations (int): Number of simulation iterations.
        random_seed (int): Seed for reproducibility.

    Returns:
        Q1_optimal (float): The total quantity to be fed into Machine 1 (includes safety stock).
        Q1_mean (float): Mean of the simulated Q1 values.
        Q1_std (float): Standard deviation of the simulated Q1 values.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Start with a multiplier of 1 for each simulation run
    multiplier = np.ones(n_simulations)

    # Multiply by (1 - waste) for each machine in the sequence
    for machine_name, waste_mean, waste_std in machine_sequence:
        w_mean = waste_mean / 100.0  # convert percentage to decimal
        w_std  = waste_std / 100.0
        waste_samples = np.random.normal(loc=w_mean, scale=w_std, size=n_simulations)
        multiplier *= (1 - waste_samples)

    # Calculate Q1 for each simulation run so that the final output equals final_demand
    Q1_samples = final_demand / multiplier

    # Compute mean and standard deviation of Q1 samples
    Q1_mean = np.mean(Q1_samples)
    Q1_std  = np.std(Q1_samples)

    # Compute safety stock using the Newsvendor model
    critical_ratio = Cu / (Cu + Co)
    Z_value = stats.norm.ppf(critical_ratio)
    safety_stock = Z_value * Q1_std

    # Total order quantity (optimal feed into Machine 1)
    Q1_optimal = Q1_mean + safety_stock

    return Q1_optimal, Q1_mean, Q1_std

# =============================================
# PARAMETERS (these are the same for all jobs)
# =============================================
Cu = 3.41       # Underage cost
Co = 0.71       # Overage cost
n_simulations = 10000

# =============================================
# READ THE INPUT FILE
# =============================================
# The Excel file is assumed to have the following columns:
# 'job_number', 'qty_ordered', 'machine_name', 'waste_mean', 'waste_std'
input_file = 'Predicted_Jobs_Grouped.xlsx'
jobs_df = pd.read_excel(input_file)

# (Optional) Strip extra spaces from column names
jobs_df.columns = [col.strip() for col in jobs_df.columns]

# =============================================
# PROCESS EACH JOB (group by job_number)
# =============================================
# We'll produce one row per machine for each job, including:
# - job_number, final_demand, Q1_optimal, Q1_mean, Q1_std (all repeated for the job)
# - machine_order (the sequence number)
# - machine_name, and the input feed into that machine.
results = []

# Group by job_number while preserving the file order
for job_number, group in jobs_df.groupby('job_number', sort=False):
    # Sort the group by the original row order (if necessary)
    group = group.sort_index()
    print(group.columns)

    # Use the first row's qty_ordered as the final demand for the job
    final_demand = group.iloc[0]['qty_ordered']

    # Build the machine sequence for this job from the rows (in order)
    machine_sequence = []
    for idx, row in group.iterrows():
        # Adjust these column names if needed based on your file
        machine_name = row['Machine Group 1']
        waste_mean = row['pred_mean']
        waste_std = row['pred_std']
        machine_sequence.append((machine_name, waste_mean, waste_std))

    # Compute Q1_optimal using the simulation for this job's final demand and machine sequence
    Q1_optimal, Q1_mean, Q1_std = compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations)

    # Now, compute the feed (input) into each machine along the sequence.
    # The input to Machine 1 is Q1_optimal.
    # The input to each subsequent machine is the output from the previous machine,
    # calculated as the previous feed multiplied by (1 - waste_mean/100).
    feed = Q1_optimal
    for order, (machine_name, waste_mean, waste_std) in enumerate(machine_sequence, start=1):
        machine_input = feed
        results.append({
            'job_number': job_number,
            'final_demand': final_demand,
            'Q1_optimal': Q1_optimal,
            'Q1_mean': Q1_mean,
            'Q1_std': Q1_std,
            'machine_order': order,
            'machine_name': machine_name,
            'machine_input': machine_input
        })
        # Update the feed for the next machine
        feed = machine_input * (1 - waste_mean / 100.0)

# =============================================
# CREATE AND EXPORT THE RESULTS
# =============================================
results_df = pd.DataFrame(results)
print(results_df)

output_file = 'Job_Machine_Quantities.xlsx'
results_df.to_excel(output_file, index=False)
print("Results saved to", output_file)
