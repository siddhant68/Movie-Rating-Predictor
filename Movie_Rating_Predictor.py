import numpy as np


# Opening cleaned reviews
with open('clean_imdb_trainX.txt', encoding='utf8') as f:
    reviews = f.readlines()
    reviews = reviews[:500]


# Creating vectorized 2d array
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(reviews).toarray()


# Adding ratings to vectorized corpus
with open('imdb_trainY.txt', encoding='utf8') as f1:
    ratings = f1.readlines()
    ratings = list(map(int, ratings))
    ratings = ratings[:500]
    
y_train = np.array(ratings)


# Getting x_test and y_test
with open('clean_imdb_testX.txt', encoding='utf8') as f3:
    test_reviews = f3.readlines()
    test_reviews = test_reviews[:50]
x_test = cv.transform(test_reviews).toarray()

with open('imdb_testY.txt', encoding='utf8') as f4:
    test_ratings = f4.readlines()
    test_ratings = list(map(int, test_ratings))
    test_ratings = test_ratings[:50]
y_test = np.array(test_ratings)


# Prior Probability
def prior_probab(x_train, y_train, label_value):
    n_rows = x_train[y_train==label_value]
    prob_val = n_rows.shape[0]/float(x_train.shape[0])
    return prob_val
prior_probab(x_train, y_train, 1)


# Conditional Probability
def cond_probab(x_train, y_train, feature_index, feature_value, label_value):
    n_rows = x_train[y_train==label_value]
    constraint_rows = n_rows[n_rows[:, feature_index]==feature_value]
    prob_val = constraint_rows.shape[0]/float(x_train.shape[0])
    return prob_val


# Calculating Classes
classes = np.unique(y_train)


# Calculating y_pred
y_pred = []
for ix in range(x_test.shape[0]):
    post_prob = []
    for jx in classes:
        likelihood = 1.0
        for kx in x_test[ix]:
            cond = cond_probab(x_train, y_train, kx, x_test[ix][kx], jx)
            likelihood *= cond
        prior = prior_probab(x_train, y_train, jx)
        posterior = likelihood * prior
        post_prob.append(posterior)
    post_prob = np.array(post_prob)
    pred_label = classes[post_prob.argmax()]
    y_pred.append(pred_label)
    

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    






