# import MeCab
# import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# 参考サイトのコピペ
# 文章をmecabで分かちがきして、名詞・動詞・形容詞の単語一覧を返す

import MeCab

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
        splited_data.append(split(datum[0]))
    return splited_data


train = np.array([
    ['あいうえお', 1],
    ['かきくけこ', 2],
    ['さしすえそ', 3],
    ['たちつてと', 4],
    ['ほげほげお', 4],
])

bodies = splited_data(train)
count_vectorizer = CountVectorizer()
feature_vectors = count_vectorizer.fit_transform(bodies)  # csr_matrix(疎行列)が返る
vocabulary = count_vectorizer.get_feature_names()

print(feature_vectors)
print(train[:,1])

estimator = LogisticRegression(C=1e5)
estimator.fit(feature_vectors, train[:,1])

X = np.array([
    ['あいうえお'],
    ['テストテスト'],
    ['さしすかきく'],
])

Xtrain = splited_data(X)
count_vectorizer_pre = CountVectorizer(
    vocabulary=vocabulary
)
feature_vectors = count_vectorizer_pre.fit_transform(Xtrain)
result = estimator.predict(feature_vectors)
print(result)
