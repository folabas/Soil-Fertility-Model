import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import networkx as nx
import matplotlib.pyplot as plt
import traceback
import joblib
import sys
import json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


# Load the trained models and the discretizer
bayesian_model_filename = 'bayesian_network_model.pkl'
discretizer_filename = 'discretizer.pkl'
random_forest_model_filename = 'random_forest_model.pkl'

bayesian_model = joblib.load(bayesian_model_filename)
discretizer = joblib.load(discretizer_filename)
random_forest_model = joblib.load(random_forest_model_filename)

# Initialize inference engine for Bayesian Network
bayesian_inference = VariableElimination(bayesian_model)

# Define trained feature order for discretizer
trained_feature_order = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

def predict_bayesian(input_data):
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([input_data])

    # Exclude features not used in discretizer
    features_for_discretizer = df_input.drop(columns=['actual_yield', 'moisture', 'soil_quality', 'temperature'])
    features_for_discretizer = features_for_discretizer[trained_feature_order]

    # Discretize the input using the fitted discretizer
    discretized = discretizer.transform(features_for_discretizer)

    # Create evidence dictionary from discretized input
    evidence = dict(zip(features_for_discretizer.columns, discretized[0].astype(int)))

    # Perform inference to predict Output
    prediction = bayesian_inference.map_query(variables=['Output'], evidence=evidence)
    fertility_value = prediction['Output']
    graded_output = grade_fertility(fertility_value)
    return graded_output if isinstance(fertility_value, str) else fertility_value

def calculate_recommendations(input_data):
    # Perform Bayesian update for fertilizer recommendation
    fertilizer_range = (50, 150)
    posterior = bayesian_update(prior, likelihood, input_data, fertilizer_range)

    # Proposal function for MCMC
    propose_new_fertilizer = lambda x: np.random.normal(x, 2)

    # Generate samples and infer optimal fertilizer
    samples = mcmc_sampling(posterior, num_samples=1000, initial_guess=100, propose_new_fertilizer=propose_new_fertilizer)
    optimal_fert, uncertainty = infer_optimal_fertilizer(samples)
    predicted_yield = predict_optimal_yield(optimal_fert, input_data)

    # Return results
    return {
        "recommended_fertilizer": f"{optimal_fert:.2f} kg ± {uncertainty:.2f}",
        "predicted_yield": f"{predicted_yield:.2f} tons/ha"
    }

def predict_random_forest(input_data):
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([input_data])

    # Only using the features the discretizer was trained on
    features_for_discretizer = df_input.drop(columns=['actual_yield', 'moisture', 'soil_quality', 'temperature'])
    features_for_discretizer = features_for_discretizer[trained_feature_order]

    # Discretize the input using the fitted discretizer
    discretized = discretizer.transform(features_for_discretizer)
    discretized_df = pd.DataFrame(discretized, columns=trained_feature_order)

    # Predict using RandomForest model
    rf_prediction = random_forest_model.predict(discretized_df)
    return rf_prediction[0]

def bayesian_update(prior, likelihood, data, fertilizer_range):
    posterior = lambda fertilizer_amount: likelihood(data, fertilizer_amount) * prior(fertilizer_amount)
    normalizer, _ = quad(posterior, fertilizer_range[0], fertilizer_range[1])
    if normalizer == 0:
        print("Normalizer is zero, check prior and likelihood functions.")
        raise ValueError("Normalizer is zero, check prior and likelihood functions.")
    return lambda fertilizer_amount: posterior(fertilizer_amount) / normalizer

def mcmc_sampling(posterior, num_samples, initial_guess, propose_new_fertilizer):
    samples = []
    current_fertilizer = initial_guess
    for _ in range(num_samples):
        proposed_fertilizer = propose_new_fertilizer(current_fertilizer)
        acceptance_probability = min(1, posterior(proposed_fertilizer) / posterior(current_fertilizer))
        if np.random.rand() < acceptance_probability:
            current_fertilizer = proposed_fertilizer
        samples.append(current_fertilizer)
    return samples

def infer_optimal_fertilizer(samples):
    optimal_fertilizer = np.mean(samples)
    uncertainty = np.std(samples)
    return optimal_fertilizer, uncertainty

def predict_optimal_yield(optimal_fertilizer, data):
    soil_quality = float(data['soil_quality'])
    moisture = float(data['moisture'])
    temperature = float(data['temperature'])
    predicted_yield = optimal_fertilizer * soil_quality * moisture * temperature
    return predicted_yield

def prior(fertilizer_amount):
    # Adjusted mean and standard deviation for better fit
    prob = norm.pdf(fertilizer_amount, loc=75, scale=15)
    return prob

def likelihood(data, fertilizer_amount):
    # Adjusted formula for predicted yield
    soil_quality = float(data['soil_quality'])
    moisture = float(data['moisture'])
    temperature = float(data['temperature'])
    predicted_yield = 0.4 * fertilizer_amount + 0.3 * soil_quality + 0.2 * moisture + 0.1 * temperature
    scale = 10  # Adjusted scale for better fit
    prob = norm.pdf(float(data['actual_yield']), loc=predicted_yield, scale=scale)
    return max(prob, 1e-10)  # Ensure non-zero probability

def grade_fertility(fertility_value):
    if isinstance(fertility_value, str):
        return fertility_value

    fertility_value = float(fertility_value)
    if 0 <= fertility_value <= 20:
        return "Low"
    elif 21 <= fertility_value <= 50:
        return "Mid"
    elif fertility_value >= 51:
        return "High"
    else:
        return "Unknown"

# Function to convert numpy types to native types
def convert_to_native_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_int(item) for item in obj]
    else:
        return obj

def main(input_json):
    input_data = json.loads(input_json)

    # Predict using both models
    bayesian_result = predict_bayesian(input_data)
    rf_result = predict_random_forest(input_data)
    results = calculate_recommendations(input_data)

    # Construct the output as a JSON object
    final_output = {
        'grade_fertility': bayesian_result,
        'predicted_yield': results['predicted_yield'],
        'recommended_fertilizer': results['recommended_fertilizer']
    }

    # Convert numpy types to native types before serializing
    final_output = convert_to_native_int(final_output)

    # Ensure that the output is valid JSON and print it
    try:
        print(json.dumps(final_output))
    except Exception as e:
        print(f"Failed to output valid JSON: {str(e)}")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("No input JSON provided.")

        input_json = sys.argv[1]
        main(input_json)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
