"""Train the two classification models and save their artifacts."""
import joblib
import pandas as pd
from preprocessing import load_dataset, prepare_classification_data
from knn_model import train_knn, save_model as save_knn
from decision_tree_model import train_decision_tree, save_model as save_tree

def main():
    df = load_dataset()
    clean, X, y, scaler, X_scaled = prepare_classification_data(df)

    knn, _, _, knn_metrics = train_knn(X_scaled, y)
    tree, _, _, tree_metrics = train_decision_tree(X, y)

    save_knn(knn, scaler, "knn_bundle.pkl")
    save_tree(tree, "decision_tree.pkl")

    metrics = pd.DataFrame([
        {"model": "KNN", **{k: v for k, v in knn_metrics.items() if k in ["accuracy", "precision", "recall", "f1"]}},
        {"model": "Decision Tree", **{k: v for k, v in tree_metrics.items() if k in ["accuracy", "precision", "recall", "f1"]}},
    ])
    metrics.to_csv("model_metrics.csv", index=False)
    print("Models trained successfully.")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
