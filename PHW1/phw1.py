import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

warnings.filterwarnings(action='ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# from google.colab import drive
# drive.mount('/content/drive')
df = pd.read_csv("data/breast-cancer-wisconsin.data", header=None)

# df = pd.read_csv("/content/drive/Shareddrives/머신러닝/breast-cancer-wisconsin.data",header=None)

df.columns = ['Sample code number', 'Clump Thickness ', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']

df = df.replace('?', np.NaN)

print(df.isnull().sum())

df.fillna(0, inplace=True)

print(df.isnull().sum())

df.drop('Sample code number', axis=1, inplace=True)

X, y = df.drop(['Class'], axis=1), df['Class']

# model, scaler, K parameter
models = ['DecisionTreeClassifier', 'LogisticRegression', 'svm.SVC']
scalers = [StandardScaler(), MinMaxScaler()]
Ks = [5, 10, 15]

# decision_tree_parameter
criterions = ['gini', 'entropy']
splitters = ['best', 'random']
max_depths = [1, 10, 100]

# logi_tree_parameter
solvers = ['lbfgs', 'sag']
max_iters = [50, 100, 200]

# svm_parameter
Cs = [0.1, 1]
gammas = [0.1, 0.3, 0.5, 1, 5]
kernels = ['rbf', 'sigmoid']
max_iters = [50, 100, 200]

# Dictionary parameter
dt_parameter = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 10, 100]
}

lg_parameter = {
    'solver': ['lbfgs', 'sag'],
    'max_iter': [50, 100, 200]
}

svm_parameter = {
    'C': [0.1, 1],
    'gamma': [0.1, 0.3, 0.5, 1, 5],
    'kernel': ['rbf', 'sigmoid'],
    'max_iter': [50, 100, 200]
}

# make parameter list
total_parameter = [scalers, Ks, criterions, splitters, max_depths, solvers, max_iters, Cs, gammas, kernels, max_iters]


def create_model(X, y, models, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    for model in models:
        if model == 'DecisionTreeClassifier':
            DT_accuracy = DT_train_model(X_train, X_test, y_train, y_test, params[0], params[1], params[2], params[3],
                                         params[4])
            accuracy = pd.concat([accuracy, DT_accuracy])
            print("\n")
        elif model == 'LogisticRegression':
            LG_accuracy = LG_train_model(X_train, X_test, y_train, y_test, params[0], params[1], params[5], params[6])
            accuracy = pd.concat([accuracy, LG_accuracy])
            print("\n")

        elif model == 'svm.SVC':
            SVM_accuracy = SVM_train_model(X_train, X_test, y_train, y_test, params[0], params[1], params[7], params[8],
                                           params[9], params[10])
            accuracy = pd.concat([accuracy, SVM_accuracy])

            print("\n")
        else:
            print("Input model Error")

    return accuracy


def DT_train_model(X_train, X_test, y_train, y_test, scalers, Ks, criterions, splitters, max_depths):
    DT_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("========================")
    print("[DecisionTreeClassifier]")
    print("========================")

    for scaler in scalers:
        for criterion in criterions:
            for splitter in splitters:
                for max_depth in max_depths:
                    for K in Ks:
                        # do use Scaler
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.fit_transform(X_test)

                        # build DecisionTreeClassifier model and fit data
                        DT = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                    random_state=42)

                        # do k-fold validation (cv=k)
                        score = cross_val_score(DT, X_train, y_train, cv=K)
                        score = np.mean(score)

                        print(
                            "DecisionTreeClassifier Average of scores : %f (scaler = %s, k = %s, criterion = %s, splitter = %s, max_depth = %s)" % (
                                score, scaler, K, criterion, splitter, max_depth))

                        data_to_insert = {'model': 'DecisionTreeClassifier', 'scaler': scaler, 'K': K,
                                          'parm': '{\'criterion\' : %s, \'splitter\' : %s, \'max_depth\' : %s}' % (
                                              criterion, splitter, max_depth), 'score': score}
                        DT_accuracy = DT_accuracy.append(data_to_insert, ignore_index=True)

    DT_accuracy = DT_accuracy.nlargest(5, 'score')
    return DT_accuracy


def LG_train_model(X_train, X_test, y_train, y_test, scalers, Ks, solvers, max_iters):
    LG_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("========================")
    print("[LogisticRegression]")
    print("========================")

    for scaler in scalers:
        for solver in solvers:
            for max_iter in max_iters:
                for K in Ks:
                    # do use Scaler
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.fit_transform(X_test)

                    # build LogisticRegression model and fit data
                    LG = LogisticRegression(solver=solver, max_iter=max_iter, random_state=42)

                    # do k-fold validation (cv=k)
                    score = cross_val_score(LG, X_train, y_train, cv=K)
                    score = np.mean(score)

                    print(
                        "LogisticRegression Average of scores : %f (scaler = %s, k = %s, solver = %s, max_iter = %s)" % (
                            score, scaler, K, solver, max_iter))

                    data_to_insert = {'model': 'LogisticRegression', 'scaler': scaler, 'K': K,
                                      'parm': '{\'solver\' : %s, \'max_iter\' : %s}' % (solver, max_iter),
                                      'score': score}
                    LG_accuracy = LG_accuracy.append(data_to_insert, ignore_index=True)

    LG_accuracy = LG_accuracy.nlargest(5, 'score')
    return LG_accuracy


def SVM_train_model(X_train, X_test, y_train, y_test, scalers, Ks, Cs, gammas, kernels, max_iters):
    SVM_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("========================")
    print("[SVM]")
    print("========================")
    for scaler in scalers:
        for C in Cs:
            for gamma in gammas:
                for kernel in kernels:
                    for max_iter in max_iters:
                        for K in Ks:
                            # do use Scaler
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.fit_transform(X_test)
                            # build SVM model and fit data
                            SVM = svm.SVC(C=C, gamma=gamma, kernel=kernel, max_iter=max_iter, random_state=42)

                            # do k-fold validation (cv=k)
                            score = cross_val_score(SVM, X_train, y_train, cv=K)

                            score = np.mean(score)

                            print(
                                "SVM Average of scores : %f (scaler = %s, k = %s, C = %s, gamma = %s, kernel = %s, max_iter = %s)" % (
                                    score, scaler, K, C, gamma, kernel, max_iter))
                            data_to_insert = {'model': 'SVM', 'scaler': scaler, 'K': K,
                                              'parm': '{\'C\' : %s, \'gamma\' : %s, \'kernel\' : %s, \'max_iter\' : %s}' % (
                                                  C, gamma, kernel, max_iter), 'score': score}
                        SVM_accuracy = SVM_accuracy.append(data_to_insert, ignore_index=True)

    SVM_accuracy = SVM_accuracy.nlargest(5, 'score')
    return SVM_accuracy


accuracy = create_model(X, y, models, total_parameter)

# sort value by score by descending order
accuracy = accuracy.sort_values(by=['score'], ascending=False)

# reset index number and restore
accuracy = accuracy.reset_index(drop=True)

print(accuracy)


# checking model parameter
def check_model(X, y, models, scalers, Ks, dt_parameter, lg_parameter, svm_parameter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    for model in models:
        if model == 'DecisionTreeClassifier':
            DT_accuracy = DT_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, dt_parameter)
            accuracy = pd.concat([accuracy, DT_accuracy])
            print("\n")
        elif model == 'LogisticRegression':
            LG_accuracy = LG_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, lg_parameter)
            accuracy = pd.concat([accuracy, LG_accuracy])
            print("\n")

        elif model == 'svm.SVC':
            SVM_accuracy = SVM_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, svm_parameter)
            accuracy = pd.concat([accuracy, SVM_accuracy])

            print("\n")
        else:
            print("Input model Error")

    return accuracy


def DT_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, dt_parameter):
    DT_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("==========================================")
    print("[DecisionTreeClassifier With GridSearchCV]")
    print("==========================================")

    for scaler in scalers:
        print("------------------------------------------")
        print("[%s]" % scaler)
        print("------------------------------------------")

        for K in Ks:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            DT = DecisionTreeClassifier(random_state=42);
            grid_DT = GridSearchCV(DT, param_grid=dt_parameter, cv=K, scoring="accuracy")
            grid_DT.fit(X_train, y_train)

            print('GridSearchCV Best parameters (using k : %s) : ' % K, grid_DT.best_params_)
            print('GridSearchCV Best accuracy : %0.6f' % grid_DT.best_score_)

            data_to_insert = {'model': 'DecisionTreeClassifier', 'scaler': scaler, 'K': K,
                              'parm': grid_DT.best_params_, 'score': grid_DT.best_score_}
            DT_accuracy = DT_accuracy.append(data_to_insert, ignore_index=True)

    DT_accuracy = DT_accuracy.nlargest(1, 'score')
    return DT_accuracy


def LG_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, lg_parameter):
    LG_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("==========================================")
    print("[LogisticRegression With GridSearchCV]")
    print("==========================================")

    for scaler in scalers:
        print("------------------------------------------")
        print("[%s]" % scaler)
        print("------------------------------------------")

        for K in Ks:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            LG = LogisticRegression(random_state=42);
            grid_LG = GridSearchCV(LG, param_grid=lg_parameter, cv=K, scoring="accuracy")
            grid_LG.fit(X_train, y_train)

            print('GridSearchCV Best parameters (using k : %s) : ' % K, grid_LG.best_params_)
            print('GridSearchCV Best accuracy : %0.6f' % grid_LG.best_score_)

            data_to_insert = {'model': 'LogisticRegression', 'scaler': scaler, 'K': K,
                              'parm': grid_LG.best_params_, 'score': grid_LG.best_score_}
            LG_accuracy = LG_accuracy.append(data_to_insert, ignore_index=True)

    LG_accuracy = LG_accuracy.nlargest(1, 'score')
    return LG_accuracy


def SVM_grid_model(X_train, X_test, y_train, y_test, scalers, Ks, svm_parameter):
    SVM_accuracy = pd.DataFrame(columns=['model', 'scaler', 'K', 'parm', 'score'])

    print("==========================================")
    print("[SVM With GridSearchCV]")
    print("==========================================")

    for scaler in scalers:
        print("------------------------------------------")
        print("[%s]" % scaler)
        print("------------------------------------------")
        for K in Ks:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            SVM = svm.SVC(random_state=42);
            grid_SVM = GridSearchCV(SVM, param_grid=svm_parameter, cv=K, scoring="accuracy")
            grid_SVM.fit(X_train, y_train)

            print('GridSearchCV Best parameters (using k : %s) : ' % K, grid_SVM.best_params_)
            print('GridSearchCV Best accuracy : %0.6f' % grid_SVM.best_score_)
            data_to_insert = {'model': 'SVM', 'scaler': scaler, 'K': K,
                              'parm': grid_SVM.best_params_, 'score': grid_SVM.best_score_}
            SVM_accuracy = SVM_accuracy.append(data_to_insert, ignore_index=True)

    SVM_accuracy = SVM_accuracy.nlargest(1, 'score')
    return SVM_accuracy


grid_accuracy = check_model(X, y, models, scalers, Ks, dt_parameter, lg_parameter, svm_parameter)

# sort value by score by descending order
grid_accuracy = grid_accuracy.sort_values(by=['score'], ascending=False)

# reset index number and restore
grid_accuracy = grid_accuracy.reset_index(drop=True)

print(grid_accuracy)

# now we know the best parameters with GridSearchCV
# Analysis the model

print("========================")
print("[DecisionTreeClassifier]")
print("========================")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = grid_accuracy[grid_accuracy['model'] == 'DecisionTreeClassifier'].scaler
scaler = scaler.array[0]
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

parm = grid_accuracy[grid_accuracy['model'] == 'DecisionTreeClassifier'].parm
parm_dic = parm.array[0]

DT = DecisionTreeClassifier(criterion=parm_dic['criterion'], max_depth=parm_dic['max_depth'],
                            splitter=parm_dic['splitter'], random_state=42);
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)

print("------------------------")
print("parameters")
print("------------------------")
print(DT.get_params())
print()

print("------------------------")
print("Accuracy")
print("------------------------")
print("Accuracy score (training) : %0.6f" % DT.score(X_train, y_train))
print("Accuracy score (testing) : %0.6f" % DT.score(X_test, y_test))  # same score -> accuracy_score(y_test, y_pred)

dt_cf = confusion_matrix(y_test, y_pred)
dt_mat = pd.DataFrame(dt_cf)
plt.figure(figsize=(5, 3))
plt.title('DecisionTreeClassifier Confusion Matrix')
sns.heatmap(dt_mat, annot=True, fmt='.1f')
plt.show()

print("---------------------")
print("Classification Report")
print("---------------------")
print(classification_report(y_test, y_pred))

print("========================")
print("[LogisticRegression]")
print("========================")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = grid_accuracy[grid_accuracy['model'] == 'LogisticRegression'].scaler
scaler = scaler.array[0]
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

parm = grid_accuracy[grid_accuracy['model'] == 'LogisticRegression'].parm
parm_dic = parm.array[0]

LG = LogisticRegression(max_iter=parm_dic['max_iter'], solver=parm_dic['solver'], random_state=42);
LG.fit(X_train, y_train)
y_pred = LG.predict(X_test)

print("------------------------")
print("parameters")
print("------------------------")
print(LG.get_params())
print()

print("------------------------")
print("Accuracy")
print("------------------------")
print("Accuracy score (training) : %0.6f" % LG.score(X_train, y_train))
print("Accuracy score (testing) : %0.6f" % LG.score(X_test, y_test))  # same score -> accuracy_score(y_test, y_pred)

lg_cf = confusion_matrix(y_test, y_pred)
lg_cf_mat = pd.DataFrame(lg_cf)
plt.figure(figsize=(5, 3))
plt.title('LogisticRegression Confusion Matrix')
sns.heatmap(lg_cf_mat, annot=True, fmt='.1f')
plt.show()

print("---------------------")
print("Classification Report")
print("---------------------")
print(classification_report(y_test, y_pred))

print("========================")
print("[SVM]")
print("========================")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = grid_accuracy[grid_accuracy['model'] == 'SVM'].scaler
scaler = scaler.array[0]
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

parm = grid_accuracy[grid_accuracy['model'] == 'SVM'].parm
parm_dic = parm.array[0]

SVM = svm.SVC(C=parm_dic['C'], gamma=parm_dic['gamma'], kernel=parm_dic['kernel'],
              max_iter=parm_dic['max_iter'], random_state=42);
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

print("------------------------")
print("parameters")
print("------------------------")
print(SVM.get_params())
print()

print("------------------------")
print("Accuracy")
print("------------------------")
print("Accuracy score (training) : %0.6f" % SVM.score(X_train, y_train))
print("Accuracy score (testing) : %0.6f" % SVM.score(X_test, y_test))  # same score -> accuracy_score(y_test, y_pred)

svm_cf = confusion_matrix(y_test, y_pred)
svm_cf_mat = pd.DataFrame(lg_cf)
plt.figure(figsize=(5, 3))
plt.title('SVM Confusion Matrix')
sns.heatmap(svm_cf_mat, annot=True, fmt='.1f')
plt.show()

print("---------------------")
print("Classification Report")
print("---------------------")
print(classification_report(y_test, y_pred))
