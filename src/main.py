import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime as dt
import numpy as np
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        # array of column names to encode
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# Read input data
data = pd.read_csv("../data/mental-health-in-tech-2016/mental-heath-in-tech-2016_20161114.csv")
print(data.head())

# Variables
OUTPUT = "Do you currently have a mental health disorder?"

clean_data = MultiColumnLabelEncoder(columns=[OUTPUT]).fit_transform(data)

print(clean_data[OUTPUT].head())



################### By Baokun##################################################### 
# Drop Obvious column and remove Maybes from the response variable
# Response Variable: "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"
df = pd.read_csv("Mental_Health_Clean.csv")

to_drop = ["Would you feel comfortable discussing a mental health disorder with your coworkers?",
           'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?',
           'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
           'Would you bring up a mental health issue with a potential employer in an interview?']
df.drop(to_drop,axis=1,inplace=True)
df = df[df["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"]!=2]

# Train Test Split, Chi-Square Test Feature Selection and Feature Importance Plot
y = df["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"]
X = df.drop(["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"],axis=1)
feature_cols=X.columns.tolist()
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
# Feature Importance
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_
feature_importance = pd.DataFrame({"Features":X.columns.tolist(),"Feature Importance":clf.feature_importances_})
feature_importance.sort_values("Feature Importance",ascending=False,inplace=True)
# Top Ten Variable (Feature Importance)
sns.barplot(x="Feature Importance",y="Features",data=feature_importance.iloc[0:12])
# 
import scipy.stats as stats
from scipy.stats import chi2_contingency
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfTabular = None
        self.dfExpected = None
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result=''
        else:
            result=colX
            print(result)
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        self._print_chisquare_result(colX, alpha)
#Feature Selection
# Chi Square Test
cT = ChiSquare(df)
testColumns = pd.concat([X_train,y_train],axis=1).columns.tolist()
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?")  
    
#  Drop unimportant Variables
col_to_drop = [
"How many employees does your company or organization have?",
"Is your employer primarily a tech company/organization?",
"Have your previous employers provided mental health benefits?",
"Were you aware of the options for mental health care provided by your previous employers?",
"Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?",
"Did your previous employers provide resources to learn more about mental health issues and how to seek help?",
"Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?",
"Do you have a family history of mental illness?",
"Have you had a mental health disorder in the past?",
"Do you currently have a mental health disorder?",
"Have you been diagnosed with a mental health condition by a medical professional?",
"Have you ever sought treatment for a mental health issue from a mental health professional?",
"If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?",
"If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?",
"What is your age?",
"What is your gender?",
"What country do you work in?"]
fs_df = df.drop(col_to_drop,axis=1)
fs_df.to_csv("newresponse_Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv")
pd.DataFrame({"Unimportant Variables":col_to_drop}).to_csv("newresponse_Dropped_Columns.csv")

# Model Fitting and Parameter Tuning with RandomForest
# Train test split
fstarget=fs_df["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"]
fsy = fstarget
fsX = fs_df.drop(["Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"],axis=1)
fsfeature_cols = fsX.columns.tolist()
fsX_train, fsX_test, fsy_train, fsy_test = train_test_split(fsX,fstarget, test_size=0.3, random_state=0)

# Getting Best Para
fsrf = RandomForestClassifier(n_estimators = 20)
# Import and Prepare Grid
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
fsrf_random = RandomizedSearchCV(estimator = fsrf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
fsrf_random.fit(fsX, fsy)
str(fsrf_random.best_params_).replace(":","=").replace("'","").replace("{","").replace("}","").replace("'","")


# Fitting with best parameter
fsrf = RandomForestClassifier(n_estimators= fsrf_random.best_params_['n_estimators'], min_samples_split= fsrf_random.best_params_['min_samples_split'], min_samples_leaf= fsrf_random.best_params_['min_samples_leaf'], max_features= fsrf_random.best_params_['max_features'], max_depth= fsrf_random.best_params_['max_depth'], bootstrap= fsrf_random.best_params_['bootstrap'])
fsrf.fit(fsX_train, fsy_train)
print(fsrf.score(fsX_test,fsy_test))
str(fsrf_random.best_params_).replace(":","=").replace("'","").replace("{","").replace("}","").replace("'","")

# Evaluate and get accuracy score, auc score, auc plot and confusion matrix
def evalClassModel(model, X, y, X_test, y_test, y_pred_class, plot=False):
    #Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    # examine the class distribution of the testing set (using a Pandas Series method)
    print('Null accuracy:\n', y_test.value_counts())
    
    # calculate the percentage of ones
    print('Percentage of ones:', y_test.mean())
    
    # calculate the percentage of zeros
    print('Percentage of zeros:',1 - y_test.mean())
    
    #Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print('Classification Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))
    
    #False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)
    
    #Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))
    
    
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))
    
    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
    
    ##########################################
    #Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    # 1D array (vector) of binary values (0, 1)
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(X_test)[0:10])

    # print the first 10 predicted probabilities for class 1
    model.predict_proba(X_test)[0:10, 1]
    
    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    if plot == True:
        # histogram of predicted probabilities
        # adjust the font size 
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)
        
        # x-axis limit from 0 to 1
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
    y_pred_prob = y_pred_prob.reshape(-1,1) 
    y_pred_class = binarize(y_pred_prob, 0.3)[0]
    
    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()
        
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
    def evaluate_threshold(threshold):
        #Sensitivity: When the actual value is positive, how often is the prediction correct?
        #Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)
    return accuracy
evalClassModel(fsrf, fsX, fsy, fsX_test, fsy_test, fsrf.predict(fsX_test), plot=True)


################### By Baokun##################################################### 

