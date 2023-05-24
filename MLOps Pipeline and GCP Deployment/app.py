##### Imports that are needed to serve the model and run the flask app #####

from flask import Flask, request
from fancyimpute import KNN, SoftImpute  # Importing necessary modules

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OrdinalEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import boxcox
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import plot_importance, XGBClassifier
from matplotlib import pyplot
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
from werkzeug.utils import secure_filename
import traceback
import operator
import six
import sys
sys.modules['sklearn.externals.six'] = six  # Setting up the environment for sklearn
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.utils import _safe_indexing
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing
from imblearn.over_sampling import SMOTE  # Importing SMOTE for oversampling

# Initializing Flask app
app = Flask(__name__)

# Flask route for status checking
@app.route('/')
def hello():
    return 'Hello Mate'

# Method to check file name extension
def check_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ["csv"]

# Flask Route to run the inference on given test data
@app.route('/eligibility', methods=['POST'])
def eligibility_check():
    if request.method == "POST":
        try:
            # Sanity check: If request does not have a file
            if 'csv' not in request.files:
                return {
                    "code": 404,
                    "msg": "Csv Not Found."
                }
            file = request.files["csv"]

            # Sanity check: If request has an empty file
            if file.filename == "":
                return {
                    "code": 404,
                    "msg": "Csv Not Found."
                }

            # Sanity check: If request has a valid CSV file
            if file and check_file(file.filename):
                filename = secure_filename(file.filename)  # Flask method to validate a file name

                # Reading the test data from the CSV file
                test = pd.read_csv(file)
                cat_cols = ['Term', 'Years in current job', 'Home Ownership', 'Purpose']  # Categorical columns

                # Running a factorizer on the categorical columns
                for c in cat_cols:
                    test[c] = pd.factorize(test[c])[0]

                # Imputing missing data with SoftImpute
                updated_test_data = pd.DataFrame(data=SoftImpute().fit_transform(test[test.columns[3:19]]),
                                                 columns=test[test.columns[3:19]].columns, index=test.index)

                # Getting the dataset ready using pd.get_dummies for dropping the dummy variables
                test_data = pd.get_dummies(updated_test_data, drop_first=True)

                # Loading the ML model from the saved file
                gbm_pickle = joblib.load('GBM_Model_version1.pkl')

                # Predicting on the test data using the loaded model
                y_pred = gbm_pickle.predict(test_data)
                y_pred = gbm_pickle.predict_proba(test_data)

                # Converting numeric predictions to textual format
                y_pred_1 = np.where(y_pred == 0, 'Loan Approved', 'Loan Rejected')

                # Assigning the prediction results to the test data received
                test['Loan Status'] = y_pred_1

                # Get the output JSON
                out_data = test.replace({np.nan: None})
                json_data = out_data.to_dict('records')
                test.to_csv('Output_Test.csv', index=False)
                test = test.to_dict('records')

                return {
                    "code": "200",
                    "msg": "Fetched Successfully",
                    "resutls": json.loads(json.dumps(json_data))
                }

            else:
                return {
                    "code": 500,
                    "msg": "Something went wrong"
                }

        except:  # Exception for system errors
            return {
                    "code": 500,
                    "msg": traceback.format_exc(),
                }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))  # Running the Flask app