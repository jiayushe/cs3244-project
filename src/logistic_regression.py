import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn import metrics

#### Import dataset
train_df = pd.read_csv('../data/Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv').dropna()
train_df.columns = map(
    lambda x: x.replace(" ", "_").replace("?", "").replace("(", "").replace(")", "").replace(",", "").replace(":", ""),
    train_df.columns
)
train_df.drop('Unnamed_0', axis=1, inplace=True)

#### Separate features and response
response = "Have_you_been_diagnosed_with_a_mental_health_condition_by_a_medical_professional"
features = train_df.columns.tolist()
features.remove(response)

#corrX = train_df.corr()
#print('correlation coefficients:')
#print(corrX[response])
#print('')

#### Train test split
X, y = train_df[features], train_df[response]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#### Model training
model = LogisticRegression(solver='lbfgs', multi_class='ovr')  
model.fit(X_train, y_train)

#coef = model.coef_
#print('regression coefficients:')
#print(coef)
#print('')

#### Model Evaluation
#y_predict = model.predic(X_test)

#acu = metrics.scorer.accuracy_score(y_test, y_predict)
#sen = metrics.scorer.recall_score(y_test, y_predict, pos_label=1)
#spe = metrics.scorer.recall_score(y_test, y_predict, pos_label=0)
#auc = metrics.scorer.roc_auc_score(y_test, y_predict)

#print('accuracy: %.2f%%' % (acu * 100))
#print('sensitivity(+): %.2f%%' % (sen * 100))
#print('specificity(-): %.2f%%' % (spe * 100))
#print('AUC score: %.2f%%' % (auc * 100))
#print('Cross-validated AUC: %.2f%%' % (cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean() * 100))

#### Test various threshold for accuracy, sensitivity and specifity 
y_pred_prob_df = pd.DataFrame(model.predict_proba(X_test))
threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
for i in threshold_list:
    print('i = {}'.format(i))
    y_pred = y_pred_prob_df.applymap(lambda x: 1 if x>i else 0)
    accuracy = metrics.scorer.accuracy_score(y_test.values.reshape(y_test.values.size,1),
        y_pred.iloc[:,1].values.reshape(y_pred.iloc[:,1].values.size,1))
    sensitivity = metrics.scorer.recall_score(y_test.values.reshape(y_test.values.size,1),
        y_pred.iloc[:,1].values.reshape(y_pred.iloc[:,1].values.size,1),
        pos_label=1)
    specificity = metrics.scorer.recall_score(y_test.values.reshape(y_test.values.size,1),
        y_pred.iloc[:,1].values.reshape(y_pred.iloc[:,1].values.size,1),
        pos_label=0)
    print('accuracy is {}'.format(accuracy))
    print('sensitivity is {}'.format(sensitivity))
    print('specificity is {}'.format(specificity))
    #print(metrics.confusion_matrix(y_test.values.reshape(y_test.values.size,1),
    #   y_pred.iloc[:,1].values.reshape(y_pred.iloc[:,1].values.size,1)))

#### Plot Precision-Recall Curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
y_prob = model.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:,1])
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[:-1], "b--", label="Precision")
plt.plot(thresholds, recall[:-1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])
