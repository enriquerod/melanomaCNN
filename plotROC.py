from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from ggplot import *



import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

# preds = clf.predict_proba(Xtest)
# skplt.plot_roc_curve(ytest, preds)
# plt.show()

X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=5)
Xtrain = X[:9000]
Xtest = X[9000:]
ytrain = y[:9000]
ytest = y[9000:]

clf = LogisticRegression()
clf.fit(Xtrain, ytrain)


# preds = clf.predict_proba(Xtest)[:,1]
preds = clf.predict_proba(Xtest)

skplt.plot_roc_curve(ytest, preds)
plt.show()


# fpr, tpr, _ = metrics.roc_curve(ytest, preds)

# df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
# ggplot(df, aes(x='fpr', y='tpr')) +\
#     geom_line() +\
#     geom_abline(linetype='dashed', slope=1,intercept=0)