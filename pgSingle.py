import streamlit as st
import math
import pandas as pd
import xgboost as xgb
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed



# Set page title
st.set_page_config(page_title="Single Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

# Title
st.markdown("## Enter Job Information:")
    
left, middle, right = st.columns([3, 5, 8], vertical_alignment="top")

with left:
    orderQty = st.number_input("Order Qty", min_value=0)
with middle:
    job1 = st.selectbox("Select Machine 1", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job2 = st.selectbox("Select Machine 2", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job3 = st.selectbox("Select Machine 3", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job4 = st.selectbox("Select Machine 4", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job5 = st.selectbox("Select Machine 5", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
with right:
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        ftOFFSET = st.selectbox("OFFSET", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        ftFLUTECODE = st.selectbox("FLUTE CODE",
                        ["0","B","BC","C","E","EB","EC","F","SBS","STRN","X"],
                        index=None,
                        placeholder="Select")
        ftCLOSURE = st.selectbox("CLOSURE TYPE", ["0"],
                        index=None,
                        placeholder="Select")
        ftCOMPONENT = st.selectbox("COMPONENT CODE",
                        ["0", "10PT","12PT","16PT","18PT","20PT","22PT","24PT","28PT","BB","BK","BM","IB","II"
                         "IK","IM","K","KI","KK","KM","M","MK","MM","PK","TB","TI","TK","TM","TSPL"],
                        index=None,
                        placeholder="Select")
        ftROTARY = st.selectbox("ROTARY DC", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        
    with col2:
        ftTESTCODE = st.number_input("TEST CODE", min_value=0, max_value=999,
                        value=None, placeholder="Enter Value")
        ftNUMBERUP = st.number_input("NUMBER UP ENTRY", min_value=0, max_value=100,
                        value=None, placeholder="Enter Value")
        ftBLANKWIDTH = st.number_input("BLANK WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftBLANKLENGTH = st.number_input("BLANK LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftITEMWIDTH = st.number_input("ITEM WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftITEMLENGTH = st.number_input("ITEM LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")


st.markdown("---")  # Horizontal line

#wasteQty = round(0.05 * orderQty)
#optQty = round(orderQty - wasteQty)
#finalQty = round(orderQty + (0.001 * orderQty))

#  Information Return
("# OPTIMAL STARTING QUANTITIES")

if st.button("Calculate", type="secondary"):
    # ---------------------------
    # Helper Functions
    # ---------------------------
    def compute_nrmse(y_true, y_pred):
        """Compute Normalized RMSE = RMSE / (max - min)"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        norm_factor = y_true.max() - y_true.min()
        return rmse / norm_factor if norm_factor != 0 else rmse

    @st.cache_data
    def load_training_data(csv_file):
        return pd.read_csv(csv_file)

    @st.cache_data
    def load_excel(file_path):
        return pd.read_excel(file_path)

    def train_single_model(i, X_train_full, y_train_full, X_test, y_test, params):
        # Split the training data into a new training and testing subset
        X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
            X_train_full, y_train_full, test_size=0.20, random_state=i
        )
        # Train a model with n_jobs=1 to avoid oversubscription in parallel mode
        model = xgb.XGBRegressor(**params)
        model.fit(X_new_train, y_new_train)

        # Calculate metrics on new train and test splits
        pred_new_train = model.predict(X_new_train)
        pred_new_test = model.predict(X_new_test)
        mse_new_train = mean_squared_error(y_new_train, pred_new_train)
        mse_new_test = mean_squared_error(y_new_test, pred_new_test)
        nrmse_new_train = compute_nrmse(y_new_train, pred_new_train)
        nrmse_new_test = compute_nrmse(y_new_test, pred_new_test)

        # Metrics on the original test set
        pred_original_test = model.predict(X_test)
        mse_original_test = mean_squared_error(y_test, pred_original_test)
        nrmse_original_test = compute_nrmse(y_test, pred_original_test)

        return (model, mse_new_train, mse_new_test, nrmse_new_train,
                nrmse_new_test, mse_original_test, nrmse_original_test)

    def train_models_parallel(n_iterations, X_train_full, y_train_full, X_test, y_test, params, max_workers=8):
        trained_models = []
        new_train_mse_list = []
        new_test_mse_list = []
        new_train_nrmse_list = []
        new_test_nrmse_list = []
        original_test_mse_list = []
        original_test_nrmse_list = []

        # Use ThreadPoolExecutor to parallelize training iterations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(train_single_model, i, X_train_full, y_train_full, X_test, y_test, params)
                       for i in range(n_iterations)]
            for future in as_completed(futures):
                (model, mse_new_train, mse_new_test, nrmse_new_train, nrmse_new_test,
                 mse_original_test, nrmse_original_test) = future.result()
                trained_models.append(model)
                new_train_mse_list.append(mse_new_train)
                new_test_mse_list.append(mse_new_test)
                new_train_nrmse_list.append(nrmse_new_train)
                new_test_nrmse_list.append(nrmse_new_test)
                original_test_mse_list.append(mse_original_test)
                original_test_nrmse_list.append(nrmse_original_test)

        # Print average metrics
        print("Average New Training NRMSE (100 iterations):", np.mean(new_train_nrmse_list))
        print("Average New Testing NRMSE (100 iterations):", np.mean(new_test_nrmse_list))
        print("Average Original Testing NRMSE (100 iterations):", np.mean(original_test_nrmse_list))
        print("\nAverage New Training MSE (100 iterations):", np.mean(new_train_mse_list))
        print("Average New Testing MSE (100 iterations):", np.mean(new_test_mse_list))
        print("Average Original Testing MSE (100 iterations):", np.mean(original_test_mse_list))

        return trained_models

    def compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations=10000, random_seed=42):
        """
        Computes the optimal Q1 using a simulation-based approach.
        """
        np.random.seed(random_seed)
        multiplier = np.ones(n_simulations)
        for machine_name, waste_mean, waste_std in machine_sequence:
            w_mean = waste_mean / 100.0
            w_std  = waste_std / 100.0
            waste_samples = np.random.normal(loc=w_mean, scale=w_std, size=n_simulations)
            multiplier *= (1 - waste_samples)
        Q1_samples = final_demand / multiplier
        Q1_mean = np.mean(Q1_samples)
        Q1_std  = np.std(Q1_samples)
        critical_ratio = Cu / (Cu + Co)
        Z_value = stats.norm.ppf(critical_ratio)
        safety_stock = Z_value * Q1_std
        Q1_optimal = Q1_mean + safety_stock
        return Q1_optimal, Q1_mean, Q1_std

    # ---------------------------
    # Phase 1: Model Training & Predictions
    # ---------------------------
    with st.spinner("Loading training data and training models, please wait..."):
        training_file = 'Grouped_Data.csv'
        df = load_training_data(training_file)
        target_col = 'Waste %'
        y = df[target_col].astype(np.float32)
        selected_features = [
            'Flute Code Grouped', 'Qty Bucket', 'Component Code Grouped',
            'Machine Group 1', 'Last Operation', 'qty_ordered',
            'number_up_entry_grouped', 'OFFSET?', 'Operation', 'Test Code'
        ]
        X = df[selected_features]
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split into training and test sets (80/20 split)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_encoded, y, test_size=0.20, random_state=42
        )

        # XGBoost parameters (set n_jobs=1 for each model when training in parallel)
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': 1
        }

        # Train models in parallel (100 iterations)
        n_iterations = 100
        trained_models = train_models_parallel(n_iterations, X_train_full, y_train_full, X_test, y_test, best_params)

        # Load new jobs file (uploaded Excel) and process for predictions
        df_jobs = pd.read_excel(uploaded_file)
        # Filter out unwanted rows
        df_jobs = df_jobs[~df_jobs['Machine Group 1'].str.strip().eq('PURCHASED BOARD/OFFSET')]
        feature_cols = [
            'Flute Code', 'Qty Bucket', 'Component Code', 'Machine Group 1',
            'Last Operation', 'qty_ordered', 'number_up_entry_1', 'OFFSET?',
            'Operation', 'Test Code'
        ]
        X_jobs = df_jobs[feature_cols].copy()
        X_jobs_encoded = pd.get_dummies(X_jobs, drop_first=True)
        # Align new jobs data with training columns
        X_jobs_encoded = X_jobs_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Predict using all trained models (using list comprehension)
        all_preds = [model.predict(X_jobs_encoded) for model in trained_models]
        all_preds = np.array(all_preds).T  # shape: (num_jobs, 100)
        df_jobs['pred_mean'] = all_preds.mean(axis=1)
        df_jobs['pred_std'] = all_preds.std(axis=1)

        # Group predictions by job_number and Machine Group 1
        group_cols = ['job_number', 'Machine Group 1']
        grouped = df_jobs.groupby(group_cols).agg({
            'pred_mean': 'mean',
            'pred_std': 'mean',
            'qty_ordered': 'first'
        }).reset_index()
        print("\nPredictions for each job-machine combination (mean & std over 100 models) with qty_ordered:")
        print(grouped)

        # Save grouped predictions
        output_file_grouped = 'Predicted_Jobs_Grouped.xlsx'
        grouped.to_excel(output_file_grouped, index=False)
        print("Grouped predictions (with qty_ordered) saved to", output_file_grouped)

    with st.spinner("Computing optimal Q1 and processing job simulations, please wait..."):
        Cu = 3.41  # Underage cost
        Co = 0.71  # Overage cost
        n_simulations = 10000

        # Reload the grouped predictions
        jobs_df = pd.read_excel(output_file_grouped)
        jobs_df.columns = [col.strip() for col in jobs_df.columns]

        results = []
        for job_number, group in jobs_df.groupby('job_number', sort=False):
            group = group.sort_index()
            final_demand = group.iloc[0]['qty_ordered']
            machine_sequence = []
            for idx, row in group.iterrows():
                machine_name = row['Machine Group 1']
                waste_mean = row['pred_mean']
                waste_std = row['pred_std']
                machine_sequence.append((machine_name, waste_mean, waste_std))
        
            Q1_optimal, Q1_mean, Q1_std = compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations)
        
            # Calculate the input for each machine in the sequence
            feed = Q1_optimal
            for order, (machine_name, waste_mean, _) in enumerate(machine_sequence, start=1):
                results.append({
                    'job_number': job_number,
                    'final_demand': final_demand,
                    'Q1_optimal': Q1_optimal,
                    'Q1_mean': Q1_mean,
                    'Q1_std': Q1_std,
                    'machine_order': order,
                    'machine_name': machine_name,
                    'machine_input': feed
                })
                feed = feed * (1 - waste_mean / 100.0)
        
        results_df = pd.DataFrame(results)
        print(results_df)
        output_file = 'Job_Machine_Quantities.xlsx'
        results_df.to_excel(output_file, index=False)
        print("Results saved to", output_file)

    #OUTPUT VIEW
    df_view = load_excel(output_file)
    if df_view.shape[1] >= 8:
        # Extract unique job numbers and allow a selection
        STjob_numbers = df_view.iloc[:, 0].dropna().unique().tolist()
        STjob_selection = st.selectbox("Select a Job Number", STjob_numbers, index=0)
        filtered_df = df_view[df_view.iloc[:, 0] == STjob_selection]
        if not filtered_df.empty:
            STmachine_input = filtered_df.iloc[:, 7].values  # Machine Input column
            STfinal_demand = filtered_df.iloc[:, 1].unique()   # Final demand column
            STmachine_name = filtered_df.iloc[:, 6].values      # Machine Name column
            final_demand_val = float(STfinal_demand[0])
            initial_input = round(float(STmachine_input[0]), 3)
            st.write(f"### Selected Job: {STjob_selection}")
            st.write(f"#### Starting Input: {initial_input}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Machine Name:**")
                st.write(STmachine_name)
            with col2:
                st.write("**Machine Input:**")
                st.write(STmachine_input)
            st.write(f"#### Final Output: {final_demand_val}")
        else:
            st.error("No data found for the selected job number.")
    else:
        st.error("Data cannot be calculated.")


# Starting Quantity
#st.write("### Total Quantity Fed In: ", optQty)

# Estimated amount of waste
#st.write("Estimated Waste: ", wasteQty)

# Finished Quantity
#st.write("Final Output Quantity: ", finalQty)