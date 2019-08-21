import pandas as pd
import xgboost
import joblib
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

# File paths
review_ratings_filepath = ('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' 
                           '/data/yelp_dataset_challenge_academic_dataset/3_neg_review.csv')

review_ratings = pd.read_csv(review_ratings_filepath, encoding="ISO-8859-1")

print(review_ratings.head(15))

# Set reviews as data type = unicode
review_ratings['clean_review'] = review_ratings['clean_review'].astype('U')
review_ratings = review_ratings[0:500000]

# Split the dataset into training, validation and test data sets
train_x, test_x, train_y, test_y = model_selection.train_test_split(review_ratings['clean_review'],
                                                                    review_ratings['stars'],
                                                                    test_size=0.2,
                                                                    random_state=1)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_x, train_y, test_size=0.25, random_state=1)


# Label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
test_y = encoder.fit_transform(test_y)

# Word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(review_ratings['clean_review'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)
test_x = tfidf_vect.transform(test_x)

xtrain_tfidf = xtrain_tfidf.todense()
xvalid_tfidf = xvalid_tfidf.todense()
test_x = test_x.todense()


# # Tuned NN engineering
# model = Sequential()
# model.add(Dense(50, input_dim=xtrain_tfidf.shape[1], kernel_initializer='uniform'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
#
# model.add(Dense(50, kernel_initializer='uniform'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
#
# model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.0001), metrics=['accuracy'])
#
# # Neural Network w/ tuned hyper params
# model.fit(xtrain_tfidf, train_y, validation_data=(xvalid_tfidf, valid_y), epochs=5, batch_size=32, verbose=1)
# predictions = model.predict(test_x)
# predictions = [int(round(p[0])) for p in predictions]
# accuracy = metrics.accuracy_score(predictions, test_y)
# print("Neural Network, WordLevel TF-IDF: ",  accuracy)
# model.save('nnmodel.h5')
#
# # Naive Bayes classification
# nb_model = naive_bayes.MultinomialNB()
# nb_model.fit(xtrain_tfidf, train_y)
# predictions = nb_model.predict(test_x)
# accuracy = metrics.accuracy_score(predictions, test_y)
# print("Naive Bayes, WordLevel TF-IDF: ", accuracy)
#
# # Logistic classification
# lg_model = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
# lg_model.fit(xtrain_tfidf, train_y)
# predictions = lg_model.predict(test_x)
# accuracy = metrics.accuracy_score(predictions, test_y)
# print("Logistic Regression, WordLevel TF-IDF: ", accuracy)

# XGB classification w/ tuned hyper params
xgb_model = xgboost.XGBClassifier(learning_rate=0.01,
                                  n_estimators=100,
                                  max_depth=3,
                                  subsample=0.8,
                                  colsample_bytree=1,
                                  gamma=1)
xgb_model.fit(xtrain_tfidf, train_y)
predictions = xgb_model.predict(xvalid_tfidf)
accuracy = metrics.accuracy_score(predictions, valid_y)
print("XGBoost, WordLevel TF-IDF: ", accuracy)
joblib.dump(xgb_model, 'xgbmodel.pkl')


# # RF classification
# rf_model = ensemble.RandomForestClassifier()
# rf_model.fit(xtrain_tfidf, train_y)
# predictions = rf_model.predict(test_x)
# accuracy = metrics.accuracy_score(predictions, test_y)
# print("Random Forest, WordLevel TF-IDF: ", accuracy)
