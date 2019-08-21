import joblib
from keras.models import load_model
import pandas as pd
from sklearn import metrics

# Filepaths and reading in csv, make sure right thing is read
test_set_filepath = ('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' 
                     '/data/yelp_dataset_challenge_academic_dataset/test_set.csv')
neural_net_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/nnmodel.h5'
naive_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/naivemodel.pkl'
log_reg_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/lgmodel.pkl'
xgb_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/xgbmodel.pkl'
rf_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/rfmodel.pkl'
tfidf_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/tfidfvect.pkl'

test_set = pd.read_csv(test_set_filepath)
print(test_set['stars'].head())

# New x and ys from test csv
test_x = test_set['clean_review'].astype('U')
test_y = test_set['stars']

# Call tf-idf, transform
tfidf_vect = joblib.load(tfidf_filepath)
test_x = tfidf_vect.transform(test_x)

# Neural Net predictions
nn = load_model(neural_net_filepath)
predictions = nn.predict(test_x)
predictions = [int(round(p[0])) for p in predictions]
accuracy = metrics.accuracy_score(predictions, test_y)
print("Neural Network, WordLevel TF-IDF: ",  accuracy)

# Naive Bayes predictions
naive = joblib.load(naive_filepath)
predictions = naive.predict(test_x)
accuracy = metrics.accuracy_score(predictions, test_y)
print("Naive Bayes, WordLevel TF-IDF: ", accuracy)

# Logistic Regression predictions
lr = joblib.load(log_reg_filepath)
predictions = lr.predict(test_x)
accuracy = metrics.accuracy_score(predictions, test_y)
print("Logistic Regression, WordLevel TF-IDF: ", accuracy)

# XGBoost predictions
xgb = joblib.load(xgb_filepath)
predictions = xgb.predict(test_x)
accuracy = metrics.accuracy_score(predictions, test_y)
print("XGBoost, WordLevel TF-IDF: ", accuracy)

# Random Forest predictions
rf = joblib.load(rf_filepath)
predictions = rf.predict(test_x)
accuracy = metrics.accuracy_score(predictions, test_y)
print("Random Forest, WordLevel TF-IDF: ", accuracy)
