import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
print("\n========== 1. LOAD DATA ==========")
DATA_PATH = "Clean_Dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Clean_Dataset.csv was not found. Put it in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# 2) PREPROCESSING
print("\n========== 2. PREPROCESSING ==========")
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Make target numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Remove rows with missing target
df.dropna(subset=["price"], inplace=True)
print("\nShape after preprocessing:", df.shape)

# 3) EDA
print("\n========== 3. EDA ==========")
# Price distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Flight Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Average price by airline
plt.figure(figsize=(9, 4))
df.groupby("airline")["price"].mean().sort_values().plot(kind="bar")
plt.title("Average Flight Price by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Price by class
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="class", y="price")
plt.title("Flight Price by Class")
plt.tight_layout()
plt.show()
# Price vs days left
sample = df.sample(min(10000, len(df)), random_state=42)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=sample, x="days_left", y="price", alpha=0.3)
plt.title("Price vs Days Left")
plt.xlabel("Days Left")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
# Price by stops
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="stops", y="price")
plt.title("Flight Price by Number of Stops")
plt.tight_layout()
plt.show()
# Correlation of numerical features
numeric_eda = ["duration", "days_left", "price"]
plt.figure(figsize=(6, 4))
sns.heatmap(df[numeric_eda].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4) FEATURE ENGINEERING

print("\n========== 4. FEATURE ENGINEERING ==========")
# Convert time categories to representative hours.
hour_map = {
    "Early_Morning": 6,
    "Morning": 9,
    "Afternoon": 14,
    "Evening": 18,
    "Night": 22,
    "Late_Night": 2,
}

df["departure_hour"] = df["departure_time"].map(hour_map)
df["arrival_hour"] = df["arrival_time"].map(hour_map)

print("Added features: departure_hour, arrival_hour")
# We don't use 'flight' as a direct feature because it behaves mainly like an identifier.
features = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "departure_hour",
    "arrival_hour",
]

X = df[features]
y = df["price"]

# 5) TRAIN / TEST SPLIT
print("\n========== 5. TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# 6) ENCODING + FEATURE SELECTION
print("\n========== 6. ENCODING + FEATURE SELECTION ==========")

# One-Hot Encoding
onehot_cols = [
    "airline",
    "source_city",
    "departure_time",
    "arrival_time",
    "destination_city",
    "class",
]

# Ordinal Encoding for stops
# zero < one < two_or_more
ordinal_cols = ["stops"]

numeric_cols = [
    "duration",
    "days_left",
    "departure_hour",
    "arrival_hour",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_cols,
        ),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[["zero", "one", "two_or_more"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ordinal_cols,
        ),
        ("numeric", "passthrough", numeric_cols),
    ],
    sparse_threshold=0,
)

# Feature selection: choose top 40 transformed features.
# It is inside the pipeline, so it learns only from training data.
K_FEATURES = 40


# 7) REGRESSION MODELS
print("\n========== 7. REGRESSION MODELS ==========")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=80,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("feature_selection", SelectKBest(f_regression, k=K_FEATURES)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])
    trained_models[name] = pipeline

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

# 8) MODEL EVALUATION / COMPARISON
print("\n========== 8. MODEL EVALUATION ==========")

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"],
).sort_values("R2", ascending=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# R2 comparison
plt.figure(figsize=(8, 4))
sns.barplot(data=results_df, x="Model", y="R2")
plt.title("Regression Model Comparison - R²")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# 9) SELECT + SAVE BEST MODEL
print("\n========== 9. SAVE BEST MODEL ==========")

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

MODEL_PATH = "flight_price_model.pkl"
joblib.dump(best_model, MODEL_PATH)

print("Best Model:", best_name)
print("Model saved as:", MODEL_PATH)


# 10) USER INPUT PREDICTION
print("\n========== 10. USER INPUT PREDICTION ==========")

print("Enter flight details below.")
print("Available airlines:", sorted(df["airline"].unique()))
print("Available cities:", sorted(df["source_city"].unique()))
print("Available departure times:", sorted(df["departure_time"].unique()))
print("Available arrival times:", sorted(df["arrival_time"].unique()))
print("Available stops:", sorted(df["stops"].unique()))
print("Available classes:", sorted(df["class"].unique()))


def get_choice(message, choices):
    while True:
        value = input(message).strip()
        if value in choices:
            return value
        print("Invalid choice. Choose from:", choices)


def get_float(message, minimum=0):
    while True:
        try:
            value = float(input(message))
            if value >= minimum:
                return value
            print(f"Value must be >= {minimum}")
        except ValueError:
            print("Please enter a number.")


def get_int(message, minimum=1):
    while True:
        try:
            value = int(input(message))
            if value >= minimum:
                return value
            print(f"Value must be >= {minimum}")
        except ValueError:
            print("Please enter an integer.")



print("\nTraining finished successfully.")
print("For a GUI, run the separate Streamlit app: streamlit run app.py")
