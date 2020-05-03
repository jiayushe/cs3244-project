import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn import metrics

train_df = pd.read_csv('../data/Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv').dropna()
train_df.columns = map(
    lambda x: x.replace(" ", "_").replace("?", "").replace("(", "").replace(")", "").replace(",", "").replace(":", ""),
    train_df.columns
)
train_df.drop('Unnamed_0', axis=1, inplace =True)

response = "Have_you_been_diagnosed_with_a_mental_health_condition_by_a_medical_professional"
features = train_df.columns.tolist()
features.remove(response)

corrX = train_df.corr()
print('correlation coefficients:')
print(corrX[response])
print('')

X, y = train_df[features], train_df[response]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression(solver='lbfgs', multi_class='ovr')  
model.fit(X_train, y_train)

coef = model.coef_
print('regression coefficients:')
print(coef)
print('')

y_predict = model.predict(X_test)

acu = metrics.scorer.accuracy_score(y_test, y_predict)
sen = metrics.scorer.recall_score(y_test, y_predict, pos_label=1)
spe = metrics.scorer.recall_score(y_test, y_predict, pos_label=0)
auc = metrics.scorer.roc_auc_score(y_test, y_predict)

print('accuracy: %.2f%%' % (acu * 100))
print('sensitivity(+): %.2f%%' % (sen * 100))
print('specificity(-): %.2f%%' % (spe * 100))
print('AUC score: %.2f%%' % (auc * 100))
