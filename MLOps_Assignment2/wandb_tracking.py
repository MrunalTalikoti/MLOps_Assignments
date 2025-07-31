

# wandb_tracking.py
import wandb
from sklearn.ensemble import RandomForestClassifier
from common_code import load_and_split, evaluate_and_plot

wandb.init(project="iris-tracking", name="rf-100-depth3")

X_train, X_test, y_train, y_test = load_and_split()

model = RandomForestClassifier(n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

acc, prec, rec = evaluate_and_plot(model, X_test, y_test)

wandb.config.update({
    "model": "RandomForest",
    "n_estimators": 100,
    "max_depth": 3
})

wandb.log({
    "accuracy": acc,
    "precision": prec,
    "recall": rec
})

wandb.save("outputs/confusion_matrix_wandb.png")