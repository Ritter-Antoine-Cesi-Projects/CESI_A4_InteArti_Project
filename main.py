import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# use TkAgg
plt.switch_backend('TkAgg')

def load_data(file_path):
    print("\n--- Loading Data ---")
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def drop_unnecessary_columns(df):
    print("\n--- Dropping Unnecessary Columns ---")
    print("Dropping unnecessary columns: 'Over18', 'StandardHours'")
    return df.drop(columns=['Over18', 'StandardHours'])

def merge_dataframes(general_data, employee_survey_data, manager_survey_data):
    print("\n--- Merging DataFrames ---")
    print("Merging dataframes on 'EmployeeID'")
    dataset = pd.merge(general_data, employee_survey_data, on='EmployeeID')
    dataset = pd.merge(dataset, manager_survey_data, on='EmployeeID')
    return dataset

def clean_data(df):
    print("\n--- Cleaning Data ---")
    df = df.dropna()
    df['Age'] = df['Age'].apply(lambda x: round(x, -1))

    # Convert categorical columns to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['Department','BusinessTravel', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'], drop_first=True)

    # Convert values to 0 and 1
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.replace({'Yes': 1, 'No': 0})
    df = df.replace({'True': 1, 'False': 0})

    # Ensure all boolean columns are converted to 0 and 1
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    print("Data cleaned.")
    return df

def load_and_prepare_data():
    print("\n--- Loading and Preparing Data ---")
    # Load data from the csv files
    general_data = load_data('Datasets/general_data.csv')
    employee_survey_data = load_data('Datasets/employee_survey_data.csv')
    manager_survey_data = load_data('Datasets/manager_survey_data.csv')
    in_time = load_data('Datasets/in_time.csv')
    out_time = load_data('Datasets/out_time.csv')

    return general_data, employee_survey_data, manager_survey_data, in_time, out_time

def process_general_data(general_data):
    print("\n--- Processing General Data ---")
    # Drop the columns that are not needed
    general_data = drop_unnecessary_columns(general_data)
    return general_data

def merge_all_data(general_data, employee_survey_data, manager_survey_data):
    print("\n--- Merging All Data ---")
    # Merge the dataframes into a single dataset using the employee_id as the key
    dataset = merge_dataframes(general_data, employee_survey_data, manager_survey_data)
    return dataset

def normalize_data(df):
    print("\n--- Normalizing Data ---")
    # Normalize the data using Min-Max scaling
    df = (df - df.min()) / (df.max() - df.min())
    print("Data normalized.")
    return df

def pipeline():
    print("\n--- Starting Data Pipeline ---")
    general_data, employee_survey_data, manager_survey_data, in_time, out_time = load_and_prepare_data()
    general_data = process_general_data(general_data)
    dataset = merge_all_data(general_data, employee_survey_data, manager_survey_data)
    dataset = clean_data(dataset)
    dataset = normalize_data(dataset)
    # Remove the employee_id column
    dataset = dataset.drop(columns='EmployeeID')
    print("Data pipeline completed.")
    return dataset

class SimplePerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        print("\n--- Training the Perceptron ---")
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            if epoch % 100 == 0:  # Print every 100 epochs
                print(f"Epoch {epoch}/{self.epochs}")
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Perceptron update rule
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

        print("Training completed.")

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        print("\n--- Making Predictions ---")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        print("Predictions made.")
        return y_predicted

def plot_weights(weights, feature_names):
    print("\n--- Plotting Weights ---")
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, weights)
    plt.xlabel('Features')
    plt.ylabel('Weight Magnitude')
    plt.title('Feature Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_weights.png')  # Sauvegarde le graphique dans un fichier
    print("Weights plot saved as 'feature_weights.png'")

def calculate_metrics(y_true, y_pred):
    print("\n--- Calculating Metrics ---")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

    return tp, fp, tn, fn

def main():
    dataset = pipeline()

    # Save the cleaned dataset
    dataset.to_csv('Datasets/cleaned_dataset.csv', index=False)

    # Split data into features and target
    X = dataset.drop(columns='Attrition').values
    y = dataset['Attrition'].values

    # Initialize and train the perceptron
    perceptron = SimplePerceptron(learning_rate=0.01, epochs=1000)
    perceptron.fit(X, y)

    # Make predictions
    predictions = perceptron.predict(X)

    # Print the number of rows and columns
    print("\n--- Dataset Information ---")
    print("The dataset has {} rows and {} columns".format(dataset.shape[0], dataset.shape[1]))

    # Print accuracy
    accuracy = np.sum(predictions == y) / len(y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate and print metrics
    tp, fp, tn, fn = calculate_metrics(y, predictions)

    # Plot the weights
    feature_names = dataset.drop(columns='Attrition').columns
    plot_weights(perceptron.weights, feature_names)

    # Identify the most influential features
    influential_features = pd.Series(perceptron.weights, index=feature_names).sort_values(key=abs, ascending=False)
    print("\n--- Most Influential Features ---")
    print(influential_features.head(10))  # Affiche les 10 caractéristiques les plus influentes

if __name__ == '__main__':
    main()