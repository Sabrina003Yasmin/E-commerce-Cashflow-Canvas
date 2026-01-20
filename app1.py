import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Ecommerce ML Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🛒 Ecommerce Customers – ML Dashboard")
st.markdown("### Feature Selection & Model Comparison")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

# ===============================
# Dataset Preview
# ===============================
st.header("📄 Dataset Preview")
st.dataframe(df.head())

features = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership"
]
target = "Yearly Amount Spent"

# creta a linerchart
chart_data=pd.DataFrame(np.random.randn(10,4),columns=['Length of Membership','Time on App','Avg. Session Length','Time on Website'])
st.line_chart(chart_data)

X = df[features]
y = df[target]

# ===============================
# Correlation-Based Feature Selection
# ===============================
st.header("🧠 Feature Selection – Correlation")

corr = df[features + [target]].corr()[target].sort_values(ascending=False)

fig, ax = plt.subplots()
sns.barplot(x=corr.values[:-1], y=corr.index[:-1], ax=ax)
ax.set_title("Feature Correlation with Target")
st.pyplot(fig)

st.write("**Insight:** Higher absolute correlation ⇒ more important feature")

# ===============================
# Train-Test Split & Scaling
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Train Models
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# ===============================
# Model Comparison Table
# ===============================
st.header("⚖️ Model Comparison")
st.dataframe(results_df)

# ===============================
# Model Comparison Visualization
# ===============================
st.header("📊 Model Performance Comparison")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=results_df.melt(id_vars="Model", value_vars=["R2 Score"]),
    x="Model",
    y="value",
    ax=ax
)
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison (Higher is Better)")
st.pyplot(fig)

# ===============================
# Best Model Selection
# ===============================
best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
st.success(f"🏆 Best Model: **{best_model_name}**")

best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

# ===============================
# Feature Importance (Coefficients)
# ===============================
st.header("📌 Feature Importance (Best Model)")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": best_model.coef_
}).sort_values(by="Coefficient", ascending=False)

fig, ax = plt.subplots()
sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# ===============================
# Prediction Section
# ===============================
st.header("🔮 Predict Yearly Amount Spent")

avg_session = st.slider("Avg. Session Length", 20.0, 40.0, 33.0)
time_app = st.slider("Time on App", 5.0, 20.0, 12.0)
time_web = st.slider("Time on Website", 30.0, 50.0, 37.0)
membership = st.slider("Length of Membership", 0.0, 10.0, 4.0)

if st.button("Predict"):
    input_data = scaler.transform(
        np.array([[avg_session, time_app, time_web, membership]])
    )
    prediction = best_model.predict(input_data)[0]
    st.success(f"💰 Predicted Yearly Amount Spent: **${prediction:.2f}**")
 