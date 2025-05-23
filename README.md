# Soil Fertility Model

This project is a soil fertility prediction system that utilizes machine learning models, including a Bayesian Network and Random Forest, to predict soil fertility based on various soil parameters.

## Requirements

Ensure you have Python installed. You can install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

## Running the Model

1. Prepare your input data in the format expected by the model. The input should include the following features:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - pH
   - Electrical Conductivity (EC)
   - Organic Carbon (OC)
   - Sulfur (S)
   - Zinc (Zn)
   - Iron (Fe)
   - Copper (Cu)
   - Manganese (Mn)
   - Boron (B)

2. Run the script with your input data:

```bash
python soil_fertility_model.py <path_to_input_json>
```

3. The model will output the predicted soil fertility category.

## Note

Ensure that the model files (`bayesian_network_model.pkl`, `discretizer.pkl`, and `random_forest_model.pkl`) are present in the same directory as the script.
#   S o i l - F e r t i l i t y - M o d e l  
 