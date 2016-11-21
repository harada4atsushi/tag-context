# import numpy as np
# from sklearn.datasets import make_multilabel_classification
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelBinarizer
#
# X1 = [
#     [1,2,3,4,5],
#     [1,1,2,2,3],
#     [2,2,3,4,5],
# ]
#
# y1 = [
#     [1,2,3],
#     [1,2],
#     [4,1],
# ]
#
# classifier = OneVsRestClassifier(SVC(class_weight='auto'))
# classifier.fit(X1, y1)
# # y2 = classifier.predict(X2)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

c = OneVsRestClassifier(SVC())
X = [[1,0,0],[0,1,0],[0,0,1],[1,0,1]]
Y = [[1],[2],[3],[1,3]]
binarizer = MultiLabelBinarizer().fit(Y)
yyy = binarizer.transform(Y)
estimator = c.fit(X,yyy)
result = estimator.predict([[1,0,0]])
print(binarizer.inverse_transform(result))
# hoge = MultiLabelBinarizer().inverse_transform(result)
# print(hoge)
