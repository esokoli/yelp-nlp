import pandas as pd

reviews_filepath = '/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master' \
                   '/data/yelp_dataset_challenge_academic_dataset/review.json'

# Shrink JSON file, old file writes to new
def shrink_json(reviews_filepath):
    s = open(reviews_filepath, encoding='utf-8')
    g = open('review2.json', "w+")

    for i in range(0, 500000):
        lol = s.readline()
        g.write(lol)


shrink_json(reviews_filepath)

reviews = pd.read_json('review2.json', lines=True)
print(list(reviews))

# Data frame to csv
reviews = reviews[['text', 'stars']]

# Multi-class labeling
rating = list(reviews['stars'])
ratings = []
for d in rating:
    if float(d) == 3:
        ratings.append(0)
    if float(d) > 3:
        ratings.append(1)
    if float(d) < 3:
        ratings.append(0)

reviews['stars'] = ratings


reviews.to_csv('/Users/edgarsokoli/PycharmProjects/Yelp/modern-nlp-in-python-master/'
               'data/yelp_dataset_challenge_academic_dataset/3_neg_review.csv')

