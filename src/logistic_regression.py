import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn import metrics as mt

train_df = pd.read_csv('../data/newresponse_Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv').dropna()
train_df.columns = map(
    lambda x: x.replace(" ", "_").replace("?", "").replace("(", "").replace(")", "").replace(",", "").replace(":", ""),
    train_df.columns
)
train_df.drop('Unnamed_0', axis=1, inplace =True)

response = "Would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisors"
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

acu = mt.scorer.accuracy_score(y_test, y_predict)
sen = mt.scorer.recall_score(y_test, y_predict, pos_label=1)
spe = mt.scorer.recall_score(y_test, y_predict, pos_label=0)

print('accuracy: %.2f%%' % (acu * 100))
print('sensitivity(+): %.2f%%' % (sen * 100))
print('specificity(-): %.2f%%' % (spe * 100))
