import os
import sys
import json
import pandas as pd
from keras.models import load_model

def load_and_rank_predictions(df, model):
    # Drop the specified columns for prediction
    X = df.drop(columns=["tableName", "key", "group", "target"])
    # Get model scores
    df['score'] = model.predict(X)
    # Rank by model score and keep the first one for each group
    df = df.sort_values(by=['group', 'score'], ascending=[True, False])
    df = df.groupby("group").first().reset_index()
    return df

def count_correct_annotations(predictions):
    # Check if the 'target' column is 1
    correct_annotations = predictions[predictions["target"] == 1]
    total_correct = correct_annotations.shape[0]
    return total_correct

def compute_performance(pred_dir, model_path, annotations_file):
    # Load the model
    model = load_model(model_path)
    
    performance = {}
    
    # Load the annotations file
    with open(annotations_file, 'r') as f:
        annotations_dict = json.load(f)
    
    for dataset, expected_annotations in annotations_dict.items():
        pred_file = os.path.join(pred_dir, f"{dataset}.csv")

        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            # Rank predictions and filter
            predictions = load_and_rank_predictions(df, model)
            
            correct_annotations = count_correct_annotations(predictions)
            precision = correct_annotations / predictions.shape[0]
            recall = correct_annotations / expected_annotations

            performance[dataset] = {"precision": precision, "recall": recall}
    
    return performance

def main():
    if len(sys.argv) != 4:
        print("Usage: python compute_performance.py <predictions_dir> <model_path> <annotations_file>")
        return

    predictions_dir = sys.argv[1]
    model_path = sys.argv[2]
    annotations_file = sys.argv[3]

    performance = compute_performance(predictions_dir, model_path, annotations_file)

    result_file = "performance_results.txt"
    with open(result_file, "w") as f:
        for dataset, metrics in performance.items():
            f.write(f"{dataset}:\n")
            f.write(f"  Precision: {metrics['precision']}\n")
            f.write(f"  Recall: {metrics['recall']}\n")

if __name__ == "__main__":
    main()