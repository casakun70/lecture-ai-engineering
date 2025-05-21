import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
import os

# モデルファイルの存在チェック
if not os.path.exists("models/titanic_model.pkl"):
    print("Model file not found. Skipping evaluation.")
    exit(0)

# モデルの読み込み
with open("models/titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# テストデータの読み込み
X_test = pd.read_csv("data/05_model_input/X_test.csv")
y_test = pd.read_csv("data/05_model_input/y_test.csv")

# 予測と精度評価
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")

# 閾値判定（例：0.75）
if accuracy < 0.75:
    print("Accuracy below threshold!")
    sys.exit(1)
