import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
#Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#Naive bayes
from sklearn.naive_bayes import GaussianNB
#Stacking
from mlxtend.classifier import StackingClassifier
train_df = pd.read_csv('E:\college\MHS\model\survey.csv')
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
train_df.drop(['comments'], axis= 1, inplace=True)
train_df.drop(['state'], axis= 1, inplace=True)
train_df.drop(['Timestamp'], axis= 1, inplace=True)
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0
# Create lists by data tpe
intFeatures = ['Age']
floatFeatures = []
stringFeatures = ['Gender','Country','State','self_employed','family_history','treatment','work_interfere','no_employees','remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical','obs_consequence','comments']
# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not identified.' % feature)
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0
# Create lists by data tpe
intFeatures = ['Age']
floatFeatures = []
# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not identified.' % feature)
#Clean 'Gender'
gender = train_df['Gender'].unique()
print(gender)
#Get rid of bullshit
stk_list = ['A little about you', 'p','Female', 'M', 'Male', 'm', 'Male-ish', 'maile', 'Trans-female',
 'Cis Female', 'F', 'something kinda male?', 'Cis Male', 'Woman', 'f', 'Mal',
 'Male (CIS)', 'queer/she/they', 'non-binary', 'Femake', 'woman', 'Make', 'Nah',
 'All', 'Enby', 'fluid', 'Genderqueer', 'Female ', 'Androgyne', 'Agender',
 'cis-female/femme', 'Guy (-ish) ^_^', 'male leaning androgynous', 'Male ',
 'Man', 'Trans woman', 'msle', 'Neuter', 'Female (trans)', 'queer',
 'Female (cis)', 'Mail', 'cis male', 'Malr' ,'femail' ,'Cis Man',
 'ostensibly male, unsure what that really means']
train_df = train_df[~train_df['Gender'].isin(stk_list)]
print(train_df['Gender'].unique())
#complete missing age with mean
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
# Fill with media() values  120
s = pd.Series(train_df['Age'])
s[s<18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s>120] = train_df['Age'].median()
train_df['Age'] = s
#Ranges of Age
train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)
#There are only 0.014% of self employed so let's change NaN to NOT self_employed
#Replace "NaN" string from defaultString
train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], "Don't know" )
#Encoding data
labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
for key, value in labelDict.items():
    print(key, value)
#Get rid of 'Country'
train_df = train_df.drop(['Country'], axis= 1)
train_df.head()
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
scaler = MinMaxScaler()
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
feature_cols1 = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = train_df[feature_cols1]
y = train_df.treatment
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.30, random_state=0)
# Create dictionaries for final graph
# Use: methodDict['Stacking'] = accuracy_score
methodDict = {}
rmseDict = ()
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
labels = []
for f in range(X.shape[1]):
    labels.append(feature_cols1[f])
def evalClassModel(model, y_test1, y_pred_class, plot=False):
    #Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(y_test1, y_pred_class))
    print('Null accuracy:n', y_test1.value_counts())
    # calculate the percentage of ones
    print('Percentage of ones:', y_test1.mean())
    # calculate the percentage of zeros
    print('Percentage of zeros:',1 - y_test1.mean())
    print('True:', y_test1.values[0:25])
    print('Pred:', y_pred_class[0:25])
    #Confusion matrix
    confusion = metrics.confusion_matrix(y_test1, y_pred_class)
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
    accuracy = metrics.accuracy_score(y_test1, y_pred_class)
    model.predict_proba(X_test1)[0:10, 1]
    y_pred_prob = model.predict_proba(X_test1)[:, 1]
    if plot == True:
        # histogram of predicted probabilities
        plt.rcParams['font.size'] = 12
        plt.hist(y_pred_prob, bins=8)
      
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')
    y_pred_prob = y_pred_prob.reshape(-1,1)
    y_pred_class = binarize(y_pred_prob, threshold=0.3)[0]

    roc_auc = metrics.roc_auc_score(y_test1, y_pred_prob)
    fpr, tpr, thresholds = metrics.roc_curve(y_test1, y_pred_prob)
    def evaluate_threshold(threshold):
       
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test1, predict_mine)

    return accuracy
def tuningCV(knn):
    k_Range = list(range(1, 31))
    k_scores = []
    for k in k_Range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
def tuningGridSerach(knn):
    
    k_Range = list(range(1, 31))

   
    param_grid = dict(n_neighbors=k_Range)

   
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    
    grid.fit(X, y)
    grid.grid_scores1_
    

    grid_mean_scores1 = [result.mean_validation_score for result in grid.grid_scores_]
def tuningRandomizedSearchCV(model, param_dist):
   
    rand1 = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    rand1.fit(X, y)
    rand1.cv_results_

    best_scores = []
    for _ in range(20):
        rand1 = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
        rand1.fit(X, y)
        best_scores.append(round(rand1.best_score_, 3))

def tuningMultParam(knn):
    
    k_Range = list(range(1, 31))
    weight_options = ['uniform', 'distance']
    
    param_grid = dict(N_neighbors=k_Range, weights=weight_options)

    
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)
def logisticRegression():
    logreg = LogisticRegression()
    logreg.fit(X_train1, y_train1)
    y_pred_class = logreg.predict(X_test1)
    accuracy_score = evalClassModel(logreg, y_test1, y_pred_class, True)
    #Data for final graph
    methodDict['Log. Regression'] = accuracy_score * 100
logisticRegression()