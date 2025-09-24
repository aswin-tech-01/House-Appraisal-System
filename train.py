import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (replace with your own)
data = {
    "bedrooms": [2, 3, 4, 3, 5],
    "bathrooms": [1, 2, 3, 2, 4],
    "sqft": [1000, 1500, 2000, 1800, 2500],
    "price": [200000, 300000, 400000, 350000, 500000]
}
df = pd.DataFrame(data)

X = df[["bedrooms", "bathrooms", "sqft"]]
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "house_model.pkl")
print("Model trained and saved as house_model.pkl")
