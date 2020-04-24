"""
To run this app, in your terminal:
> python bike_regression_api.py
"""
import connexion
from sklearn.externals import joblib
from train_model1 import X, station_dic
# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
rf = joblib.load('./model/bike_classifier.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict(1 ,1, 1)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

def predict(station_id, month, week):
    for key in station_dic:
        if(key == station_id):
            station1 = station_dic[key]
    print(station1)
    user_input = X[(X['from_station_id'] == station1) & (X['month'] == month) & (X['week_of_month'] == week)][0:1]
    if len(user_input) == 1:
        predicted_value = rf.predict(user_input)
        return {"prediction" : predicted_value[0]}
    else:
        return {"prediction" : 'invalid input'}

# Read the API definition for our service from the yaml file
app.add_api("bike_regression_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
