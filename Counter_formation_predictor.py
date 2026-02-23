import joblib
import pandas as pd
import warnings
import Configurations as CF

warnings.filterwarnings('ignore')

def predict_counter_strategy(structure, shape, mode):
    """
    Takes explicit tactical features and predicts the best counter-formations.
    """
    try:
        tactical_ai = joblib.load(CF.COUNTER_FORMATION_PREDICTOR_PATH)
        tactical_encoder = joblib.load(CF.COUNTER_FORMATION_TARGET_LABEL_ENCODER_PATH)
        AI_READY = True
    except FileNotFoundError:
        print("Warning: ML model not found. Tactical recommendations disabled.")
        AI_READY = False

    # Maps the ML model's primary prediction to the original dataset's alternatives
    tactical_alternatives = {
        "4-5-1": ["4-1-3-2", "4-4-2", "4-3-3", "4-2-3-1"],
        "3-4-2-1": ["3-5-2", "4-4-1-1"],
        "3-5-2": ["4-3-3"],
        "3-4-3": ["5-3-2"],
        "4-2-3-1": ["5-3-2", "4-3-3"],
        "4-3-1-2": ["4-2-3-1", "5-3-2"],
        "4-3-3": ["4-4-2"],
        "3-5-1": ["3-5-1-1"]
    }

    if not AI_READY:
        return "N/A", []

    # 1. Handle empty inputs to match our training preprocessing
    if not structure: structure = "Unknown"
    if not shape: shape = "Unknown"
    if not mode: mode = "Unknown"

    # 2. Package into a Pandas DataFrame exactly like the training data
    X_input = pd.DataFrame({
        "Structure": [str(structure)],
        "Shape": [str(shape)],
        "Mode": [str(mode)]
    })

    # 3. Predict the encoded target using the Random Forest
    encoded_pred = tactical_ai.predict(X_input)
    
    # 4. Decode the integer back to a string (e.g., '4-5-1')
    primary_counter = tactical_encoder.inverse_transform(encoded_pred)[0]

    # 5. Look up the alternatives
    alternatives = tactical_alternatives.get(primary_counter, [])

    return primary_counter, alternatives