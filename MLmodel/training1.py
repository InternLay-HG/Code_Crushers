import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(total_minutes):
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def read_time_data(file_path):
    df = pd.read_csv(file_path)
    return df['time'].tolist()

def write_predicted_time(file_path, predicted_time):
    predicted_df = pd.DataFrame({'time': [predicted_time]})
    predicted_df.to_csv(file_path, mode='a', header=False, index=False)

def predict_next_time(file_path):
    time_data = read_time_data(file_path)
    total_minutes = [time_to_minutes(t) for t in time_data]
    differences = [total_minutes[i] - total_minutes[i - 1] for i in range(1, len(total_minutes))]
    average_difference = np.mean(differences)
    X = np.array(total_minutes[:-1]).reshape(-1, 1)
    y = np.array(total_minutes[1:])
    model = LinearRegression()
    model.fit(X, y)
    last_time = total_minutes[-1]
    predicted_next_time = model.predict(np.array([[last_time]]))[0]
    predicted_time_str = minutes_to_time(int(predicted_next_time))
    print("Input Times:", time_data)
    print("Differences (in minutes):", differences)
    print("Average Difference (in minutes):", average_difference)
    print("Predicted Next Time:", predicted_time_str)
    write_predicted_time(file_path, predicted_time_str)

if __name__ == "__main__":
    csv_file_path = r'C:\qletFrontend\ML model\time_data.csv'
    predict_next_time(csv_file_path)