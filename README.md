# 📡 Telco Customer Churn Prediction & Web App

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-009688.svg?logo=fastapi)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0+-38B2AC.svg?logo=tailwind-css)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.2-1272B5.svg?logo=xgboost)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

A complete end-to-end Machine Learning pipeline and web application that predicts customer churn in the telecommunications industry. This project features a trained XGBoost classifier exposed via a high-performance **FastAPI** backend and an intuitive, beautifully styled **Tailwind CSS** frontend.

---

## 🌟 Features

- **Real-Time Predictions**: Get instantaneous churn probabilities via a REST API.
- **Modern Web UI**: Clean, responsive, and user-friendly interface built with Tailwind CSS.
- **Robust Preprocessing Pipeline**: Automatically handles missing values, scales numeric features, computes domain-specific features, and one-hot encodes categorical data exactly as trained in the model.
- **High Accuracy ML Model**: Built using XGBoost and optimized with Hyperparameter Tuning and SMOTE for handling class imbalances.

---

## 🛠️ Tech Stack

- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy, Imbalanced-Learn
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend UI**: HTML5, Jinja2 Templates, Tailwind CSS (via CDN)
- **Deployment & Serialization**: Joblib

---

## 📂 Project Structure

```bash
├── main.py                  # FastAPI server & route definitions
├── preprocess.py            # Data transformation & scaling pipeline
├── churn_model.pkl          # Pre-trained XGBoost classification model
├── requirements.txt         # Project dependencies
├── templates/
│   └── index.html           # Tailwind CSS Frontend UI
├── Churn_prediction.ipynb   # Jupyter Notebook containing EDA & Model Training
└── README.md                # Project documentation
```

---

## 🚀 Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/ShakeelRana624/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server
```bash
uvicorn main:app --reload
```

The server will start on `http://127.0.0.1:8000`. 
*Note: The `--reload` flag allows the server to automatically restart when code changes are detected.*

---

## 💻 Usage

### Web Interface
1. Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. Fill out the customer demographics, services, and billing details in the form.
3. Click **Predict Churn** to see if the customer is likely to stay or leave, along with their churn probability.

### API Endpoint (`/predict`)
You can also interact with the model directly by sending a POST request with a JSON payload to `/predict`.

**Example Request:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 34,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Mailed check",
  "MonthlyCharges": 56.95,
  "TotalCharges": 1889.5
}'
```

**Example Response:**
```json
{
  "churn": "No",
  "probability": 14.5
}
```

*Interactive API documentation is available out-of-the-box at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

---

## 🧠 Model Training (Jupyter Notebook)
The `Churn_prediction.ipynb` notebook contains the complete lifecycle of the model:
1. **Exploratory Data Analysis (EDA)** & Data Cleaning.
2. **Feature Engineering**: Creating computed features like `TotalServices` and `ChargeDifference`.
3. **Imbalance Handling**: Using **SMOTE** to balance the churning and non-churning customer data.
4. **Model Selection**: Comparing Logistic Regression, Decision Trees, Random Forest, SVM, and XGBoost.
5. **Hyperparameter Tuning**: Fine-tuning the best model for maximum precision and recall.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ShakeelRana624/Customer-Churn-Prediction/issues).

## 📬 Contact
For any queries or collaborations, feel free to reach out:
- **Email:** shakeelrana6240@gmail.com
- **GitHub:** [ShakeelRana624](https://github.com/ShakeelRana624)

## 📄 License
This project is licensed under the [Apache-2.0 license](https://github.com/ShakeelRana624/Customer-Churn-Prediction#).
