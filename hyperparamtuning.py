import pandas as pd
import talos
from xgboost import XGBClassifier
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


# File path
review_ratings_filepath = ('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' 
                           '/data/yelp_dataset_challenge_academic_dataset/3_neg_review.csv')

review_ratings = pd.read_csv(review_ratings_filepath, encoding="ISO-8859-1")

print(review_ratings.head(5))
print(review_ratings.dtypes)


# Set reviews as data type = unicode
review_ratings['clean_review'] = review_ratings['clean_review'].astype('U')
review_ratings = review_ratings[0:500000]

# Split the dataset into training, validation and test data sets (60/20/20)
train_x, test_x, train_y, test_y = train_test_split(review_ratings['clean_review'], review_ratings['stars'],
                                                    test_size=0.2, random_state=1)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)

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


# Tuned NN engineering
def model_nn(xtrain_tfidf, train_y, xvalid_tfidf, valid_y, params):
    # create layers
    model = Sequential()
    model.add(Dense(params['input_layer'], input_dim=xtrain_tfidf.shape[1],
                    kernel_initializer=params['kernel_initializer']))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(params['second_layer'], kernel_initializer=params['kernel_initializer']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Activation('relu'))

    model.add(Dense(params['output_layer'], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))

    model.compile(loss=binary_crossentropy, optimizer=params['optimizer'](lr=params['learning_rate']),
                  metrics=['accuracy'])

    out = model.fit(xtrain_tfidf, train_y, validation_data=[xvalid_tfidf, valid_y],
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=1)
    return out, model


# Neural network hyper-parameter tuning
p = {'input_layer': [50, 100, 200],
     'second_layer': [50, 100, 200],
     'dropout': [0.5, 0.65],
     'output_layer': [1],
     'batch_size': [32, 128],
     'learning_rate': [0.00001, 0.0001],
     'epochs': [5],
     'kernel_initializer': ['uniform'],
     'activation': ['relu'],
     'optimizer': [Adam],
     'last_activation': ['sigmoid']}

t = talos.Scan(x=xtrain_tfidf,
               y=train_y,
               model=model_nn,
               params=p,
               grid_downsample=1,
               dataset_name='yelp',
               experiment_no='9')

r = talos.Reporting('yelp_.csv')
r.best_params()

# XGB classification
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'n_estimators': [50, 100, 200, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier(learning_rate=0.02,
                    silent=True,
                    nthread=2)

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds,
                      shuffle=True,
                      random_state=1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   cv=skf.split(xtrain_tfidf, train_y),
                                   verbose=1,
                                   random_state=1001)
random_search.fit(xtrain_tfidf.tocsc(), train_y)
print(random_search.best_params_)

xgb_model = xgb
xgb_model.fit(xtrain_tfidf.tocsc(), train_y)
predictions = xgb_model.predict(xvalid_tfidf.tocsc())
accuracy = metrics.accuracy_score(predictions, valid_y)
print("XGBoost, WordLevel TF-IDF: ", accuracy)
