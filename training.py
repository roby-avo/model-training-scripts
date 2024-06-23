import numpy as np
import pandas as pd
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 32768

def train_model(training_file, columns_to_exclude, target_column, json_config_file):
    data = pd.read_csv(training_file)

    # Read the JSON configuration file
    with open(json_config_file, 'r') as file:
        config = json.load(file)
    
    # Set specified columns to 0
    columns_to_zero = config.get('columns_to_zero', [])
    data[columns_to_zero] = 0

    # Preprocess your data: remove unwanted columns
    X = data.drop(columns=columns_to_exclude + [target_column])
    y = data[target_column]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    categorical_y = to_categorical(encoded_y)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, categorical_y, test_size=0.2, random_state=42)
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(encoded_y), y=encoded_y)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Define the Neural Network structure with L2 regularization
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')  # Output layer neurons = number of classes
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the Neural Network with early stopping and class weights
    model.fit(X_train, y_train, epochs=100, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict(X_train, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate the classification report
    report = classification_report(np.argmax(y_train, axis=1), predictions)
    
    # Print out the classification report to a file
    report_filename = f"report_{training_file.split('/')[-1].split('.')[0]}.txt"
    with open(report_filename, "w") as f:
        f.write(report)
    
    # Save the trained model
    model_filename = f"model_{training_file.split('/')[-1].split('.')[0]}.h5"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    print(f"Classification report saved as {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='Train a neural network model on the provided dataset.')
    parser.add_argument('training_file', type=str, help='Path to the training CSV file.')
    parser.add_argument('--columns_to_exclude', type=str, nargs='+', default=[], help='List of columns to exclude from training.')
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--json_config_file', type=str, required=True, help='Path to the JSON configuration file.')

    args = parser.parse_args()
    train_model(args.training_file, args.columns_to_exclude, args.target_column, args.json_config_file)

if __name__ == '__main__':
    main()