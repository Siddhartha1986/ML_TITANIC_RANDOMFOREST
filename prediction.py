import joblib

def PREDICT(data):
    clf = joblib.load("output_models/new_rfc_model.sav")
    return clf.predict(data)