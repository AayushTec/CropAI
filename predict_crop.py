import pandas as pd
import numpy as np
import pickle

def load_model_and_scaler():
    with open('crop_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('crop_prediction_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    model, scaler = load_model_and_scaler()
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

def get_user_input():
    print("\nEnter the following details:")
    try:
        N = float(input("Nitrogen content in soil (N): "))
        P = float(input("Phosphorus content in soil (P): "))
        K = float(input("Potassium content in soil (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH value: "))
        rainfall = float(input("Rainfall (mm): "))
        return N, P, K, temperature, humidity, ph, rainfall
    except ValueError:
        print("Please enter valid numerical values!")
        return None

def main():
    print("Welcome to Crop Prediction System")
    print("=================================")
    
    while True:
        values = get_user_input()
        if values is None:
            continue
            
        try:
            crop = predict_crop(*values)
            print("\nPrediction Results:")
            print("-------------------")
            print(f"Recommended crop: {crop}")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
        
        choice = input("\nWould you like to make another prediction? (y/n): ")
        if choice.lower() != 'y':
            break
    
    print("\nThank you for using the Crop Prediction System!")

if __name__ == "__main__":
    main()