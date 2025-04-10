import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Create sample data
data = {
    'Brand': ['Peugeot', 'Renault', 'Dacia', 'Volkswagen', 'Toyota'] * 100,
    'Model': ['208', 'Clio', 'Logan', 'Golf', 'Yaris'] * 100,
    'Year': np.random.randint(2010, 2024, 500),
    'Damaged_Part': ['Pare-brise arriere', 'Porte arriere gauche', 'Aile arriere droit', 'Capot', 'Grille'] * 100,
    'Damage_Severity': np.random.randint(1, 4, 500),
    'Damage_Cost_MAD': np.random.randint(500, 5000, 500)
}

df = pd.DataFrame(data)

# Feature selection
features = ['Brand', 'Model', 'Year', 'Damaged_Part', 'Damage_Severity']
target = 'Damage_Cost_MAD'

# Split dataset into features and target
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for categorical features
categorical_features = ['Brand', 'Model', 'Damaged_Part']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Transform data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Train Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42
)

model.fit(X_train_encoded, y_train)

# Save the model and preprocessor
joblib.dump(model, 'gradient_boosting_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model and preprocessor saved successfully!") 