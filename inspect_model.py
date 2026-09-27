import joblib

model = joblib.load('c:/Users/shake/OneDrive/Desktop/Customer-Churn-Perdiction/churn_model.pkl')
try:
    print(model.feature_names_in_)
except:
    try:
        print(model.get_booster().feature_names)
    except:
        print("Could not get feature names")
