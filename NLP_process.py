import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd


# File paths
review_ratings_filepath = ('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' 
                           '/data/yelp_dataset_challenge_academic_dataset/3_neg_review.csv')

review_ratings = pd.read_csv(review_ratings_filepath, encoding="ISO-8859-1")

print(review_ratings.head(5))
print(review_ratings.dtypes)

# Convert to list for data processing
# Set reviews as data type = string
review_ratings['clean_review'] = review_ratings['clean_review'].astype('str')
review_list = list(review_ratings['clean_review'])
print(review_list)

# Lowercase processing
review_list = [k.lower() for k in review_list]
print(review_list[0])

# Punctuation removal
review_list = [''.join(c for c in s if c not in string.punctuation) for s in review_list]
print(review_list[0])

# Whitespace removal
review_list = [x.strip(' ') for x in review_list]
print(review_list[0])

# List back to column in data frame
review_ratings['clean_review'] = review_list

# Remove stopwords
stop = stopwords.words('english')
review_ratings['clean_review'] = review_ratings['clean_review'].apply(lambda x: ' '.join([word for word in x.split()
                                                                                          if word not in (stop)]))
# Lemmatization
review_lemma = list(review_ratings['clean_review'])
lmtzr = WordNetLemmatizer()
review_lemma = [lmtzr.lemmatize(x) for x in review_lemma]
print(review_lemma[0])

# List back to column in data frame
review_ratings['clean_review'] = review_lemma

review_ratings.to_csv('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' 
                      '/data/yelp_dataset_challenge_academic_dataset/3_neg_review.csv')
