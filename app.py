from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
from io import BytesIO
import base64
from weather import weather_fetch, aqi_fetch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Assume you have trained all six models and saved them
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def main():
    if request.method == 'POST':
        city = request.form['city']
        weather_data = weather_fetch(city)
        
        if weather_data:
            temp_max, temp_min, avg_temp, pressure, humidity, visibility, wind, lat, lon = weather_data
            lat = str(lat)
            lon = str(lon)
            aqi = aqi_fetch(lat, lon)
            data = [[temp_max, temp_min, avg_temp, pressure, humidity, visibility, wind]]

            # Load all models
            dt_classifier = joblib.load("decision_tree_model.joblib")
            rf_classifier = joblib.load("random_forest_model.joblib")
            xgb_classifier = joblib.load("xgboost_model.joblib")

            # Make predictions
            dt_pm = np.round(dt_classifier.predict(data)[0], 2)
            rf_pm = np.round(rf_classifier.predict(data)[0], 2)
            xgb_pm = np.round(xgb_classifier.predict(data)[0], 2)

            return render_template("index.html", 
                                   dt_pm=dt_pm,
                                   rf_pm=rf_pm,
                                   xgb_pm=xgb_pm,
                                   city=city, 
                                   aqi=aqi)
        else:
            title = 'Oops!! Something Went Wrong.'
            return render_template('index.html', title=title)


@app.route('/results')
def results():
    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=1000, n_features=7, n_classes=2, n_clusters_per_class=1, random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)

    # Make predictions
    dt_predictions = dt_classifier.predict(X_test)
    rf_predictions = rf_classifier.predict(X_test)
    xgb_predictions = xgb_classifier.predict(X_test)

    # Calculate classification metrics
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
    dt_classification_report = classification_report(y_test, dt_predictions)

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
    rf_classification_report = classification_report(y_test, rf_predictions)

    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predictions)
    xgb_classification_report = classification_report(y_test, xgb_predictions)

    # Plotting Decision Tree
    plt.figure(figsize=(12, 6))
    from sklearn.tree import plot_tree
    plot_tree(dt_classifier, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])], class_names=['Class 0', 'Class 1'])
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    dt_tree_plot = base64.b64encode(img_stream.getvalue()).decode('utf-8')

    return render_template('results.html', 
                           dt_accuracy=dt_accuracy,
                           dt_conf_matrix=dt_conf_matrix,
                           dt_classification_report=dt_classification_report,
                           rf_accuracy=rf_accuracy,
                           rf_conf_matrix=rf_conf_matrix,
                           rf_classification_report=rf_classification_report,
                           xgb_accuracy=xgb_accuracy,
                           xgb_conf_matrix=xgb_conf_matrix,
                           xgb_classification_report=xgb_classification_report,
                           dt_tree_plot=dt_tree_plot)

# Error handling
@app.errorhandler(403)
def forbidden(error):
    return "Forbidden: You don't have permission to access this resource.", 403

if __name__ == "__main__":
    app.run(debug=True)
