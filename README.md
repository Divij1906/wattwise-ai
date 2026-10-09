⚡ WATTWISE AI – Smart Energy Prediction Dashboard

WATTWISE AI is a machine-learning-powered web application that estimates household electricity consumption and electricity bills using appliance usage patterns and household information.

The project combines machine learning, data visualization, and a modern Streamlit dashboard to make energy consumption easier to understand.

🚀 Live Demo

Live App: "Open WATTWISE AI"(https://wattwise-energy.streamlit.app/)

✨ Features

- Electricity Consumption Prediction: Estimates household electricity usage.
- Bill Estimation: Calculates an estimated electricity bill using an illustrative tariff structure.
- Bill Category Classification: Classifies predicted usage into categories.
- Usage Clustering: Uses K-Means to identify household usage groups.
- Prediction History: Stores and displays previous predictions using SQLite.
- Interactive Dashboard: Presents metrics, charts, and appliance-related insights.
- Energy-Saving Tips: Provides practical suggestions for reducing electricity consumption.

🧠 Machine Learning Models

The project includes:

- Linear Regression: Predicts electricity consumption.
- Logistic Regression: Classifies predicted usage.
- K-Means Clustering: Groups usage patterns.
- StandardScaler: Scales input features before model prediction.

The application loads pre-trained models and supporting files using Joblib.

🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- SQLite
- Matplotlib/Streamlit chart components, where applicable

⚙️ Run Locally

1. Clone the repository

git clone Divij1906
cd wattwise-ai

2. Install dependencies

python -m pip install -r requirements.txt

3. Start the application

python -m streamlit run app.py

The application will display a local URL in the terminal. Open that URL in your browser.

📁 Project Structure

wattwise-ai/
├── app.py
├── database.py
├── requirements.txt
├── .gitignore
├── scaler.pkl
├── linear_regression_model.pkl
├── logistic_regression_model.pkl
├── kmeans_model.pkl
├── pca_model.pkl
├── cluster_labels.pkl
├── feature_columns.pkl
└── median_units.pkl

🔄 How It Works

1. The user enters household and appliance usage details.
2. The application prepares the input features.
3. The pre-trained machine learning models generate predictions and usage classifications.
4. The application estimates the electricity bill.
5. Results and visualizations are displayed on the dashboard.
6. Prediction records are stored in a local SQLite database.

⚠️ Limitations

- Predictions depend on the quality and representativeness of the model's training data.
- Estimated electricity bills may differ from actual bills because real tariffs, taxes, fixed charges, and other adjustments can vary.
- The dashboard is an educational project and should not be treated as an official billing tool.
- SQLite history on a cloud deployment may be temporary because the hosting environment does not guarantee persistent local storage.
- The displayed predictions should not be interpreted as proof of model accuracy without a separate evaluation using suitable test data.

🎯 Project Objective

The objective of WATTWISE AI is to demonstrate how machine learning can be integrated into a practical web dashboard for electricity consumption estimation, usage analysis, and energy awareness.

👨‍💻 Built With

Developed as a student machine learning project using Python and Streamlit.
