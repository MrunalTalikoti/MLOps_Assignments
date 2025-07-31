
    # mlflow_tracking.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from common_code import load_and_split, evaluate_and_plot

X_train, X_test, y_train, y_test = load_and_split()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    acc, prec, rec = evaluate_and_plot(model, X_test, y_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("outputs/confusion_matrix.png")