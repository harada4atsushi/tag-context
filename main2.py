import MeCab
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def split(text):
    tagger = MeCab.Tagger()
    # text = text.encode("utf-8")
    node = tagger.parseToNode(text)
    word_list = []
    while node:
        # logger.debug(node.feature)
        features = node.feature.split(",")
        pos = features[0]
        # if pos in ["名詞", "動詞", "形容詞", "感動詞", "助動詞", "副詞"]:
        if pos in ["名詞", "動詞", "形容詞", "感動詞", "副詞"]:
            if pos == '名詞' and features[1] == '非自立':
                node = node.next
                continue
            lemma = node.feature.split(",")[6]  #.decode("utf-8")
            if lemma == 'ある':
                node = node.next
                continue

            if lemma == "*":
                lemma = node.surface  #.decode("utf-8")

            word_list.append(lemma)
        node = node.next
    return " ".join(word_list)

def parse():
    lines = []
    for line in open('data.csv', 'r'):
        arr = line.split("\t")
        lines.append(arr)
    return lines



def splited_data(train):
    splited_data = []
    for datum in train:
        splited_data.append(split(datum[8]))
    return splited_data

# [greeting, pc, buy, fail, question, password, key, forget]
train = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 2],
    [0, 1, 1, 0, 1, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 1, 0, 1, 4],
    [0, 0, 0, 0, 0, 0, 1, 1, 5],
])


# bodies = splited_data(train)
# print(bodies)
# count_vectorizer = CountVectorizer()
# feature_vectors = count_vectorizer.fit_transform(bodies)  # csr_matrix(疎行列)が返る
# feature_arr = feature_vectors.toarray()
# vocabulary = count_vectorizer.get_feature_names()

# feature_learn = np.c_[train[:,0:5], feature_arr]
# feature_learn = feature_learn.astype(float)
label_learn = train[:,-1]

# ロジスティック回帰
# estimator = LogisticRegression(C=1e5)
# estimator.fit(feature_learn, label_learn)

# SVM
# svm_tuned_parameters = [
#     {
#         'kernel': ['rbf', 'linear'],
#         'gamma': [2**n for n in range(-15, 3)],
#         'C': [2**n for n in range(-5, 15)]
#     }
# ]
# gscv = GridSearchCV(
#     SVC(probability=True),
#     svm_tuned_parameters,
#     cv=2,
#     n_jobs = 1,
#     verbose = 3
# )
# gscv.fit(feature_learn, label_learn)
# estimator = gscv.best_estimator_

# ナイーブベイズ
gnb = MultinomialNB()
estimator = gnb.fit(train[:,:-1], label_learn)

# [greeting, pc, buy, fail, question, password, key, forget]
X = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],  # => 1
    [0, 1, 0, 0, 0, 0, 0, 0],  # => fail
    [0, 1, 1, 0, 0, 0, 0, 0],  # => 3
    [0, 1, 1, 0, 1, 0, 0, 0],  # => 3
    [0, 0, 0, 0, 1, 0, 0, 0],  # => fail
    [0, 0, 1, 0, 0, 0, 0, 0],  # => fail
    [1, 0, 0, 0, 0, 0, 0, 0],  # => fail
])

# bodies = splited_data(X)
# count_vectorizer_pre = CountVectorizer(
#     vocabulary=vocabulary
# )
# feature_vectors = count_vectorizer_pre.fit_transform(bodies)
# feature_arr_predict = feature_vectors.toarray()
# feature = np.c_[X[:,0:5], feature_arr_predict]
# feature = feature.astype(float)
# print(feature)
result = estimator.predict(X)
print(result)

result_proba = estimator.predict_proba(X)
print(result_proba)
