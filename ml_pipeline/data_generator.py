import pandas as pd
import numpy as np
import os

def generate_dummy_data(num_samples=1000, output_path="data/workout_data.csv"):
    np.random.seed(42)
    
    # 6 Inputs: Age, Gender, Target Weight, Target Fat, Target Muscle, Purpose
    age = np.random.randint(18, 65, num_samples)
    gender = np.random.randint(0, 2, num_samples) # 0: Male, 1: Female
    target_weight = np.random.normal(70, 15, num_samples).round(1)
    target_fat = np.random.normal(15, 5, num_samples).round(1)
    target_muscle = np.random.normal(30, 8, num_samples).round(1)
    purpose = np.random.randint(0, 3, num_samples) # 0: Diet, 1: Hypertrophy, 2: Strength

    
    # Target Output (Program classes)
    # 0: Cardio/Weight Loss, 1: Hypertrophy, 2: Endurance, 3: Mobility/Rehab, 4: Maintenance
    program = np.random.randint(0, 5, num_samples)
    
    # Creating some artificial correlations so the model learns relationships
    for i in range(num_samples):
        if purpose[i] == 1 or (target_fat[i] < 12 and target_muscle[i] > 35):
            program[i] = 1 # Hypertrophy
        elif purpose[i] == 0 or target_fat[i] > 25:
            program[i] = 0 # Cardio
        elif purpose[i] == 2:
            program[i] = 2 # Endurance / Strength
        elif age[i] > 55:
            program[i] = 3 # Mobility

            
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "target_weight": target_weight,
        "target_fat": target_fat,
        "target_muscle": target_muscle,
        "purpose": purpose,
        "program": program
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples at {output_path}")

if __name__ == "__main__":
    generate_dummy_data()
