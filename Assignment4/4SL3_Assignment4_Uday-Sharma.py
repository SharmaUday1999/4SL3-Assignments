import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import threading
from multiprocessing import Process

RANDOM_SEED = 9248
DATA_FILEPATH = 'spambase.data'

min_dt_err = 0

def getErrors(rangeGiven,errFunc):
    min_err = math.inf
    best_num = 0
    errVals = []
    x_val = []
    x_val2 = []

    for num in range(rangeGiven[0],rangeGiven[1]+1,rangeGiven[2]):
        #print(num, " of ", rangeGiven[1])
        err,cv_err = errFunc(num)
        x_val.append(num)
        errVals.append(err)
        x_val2.append(cv_err)

        if err < min_err:
            best_num = num
            min_err = err
            
    return best_num, x_val, errVals, x_val2, min_err


#decision tree
def decisionTree(max_leaves):
    tree = DecisionTreeClassifier(random_state=RANDOM_SEED, max_leaf_nodes=max_leaves).fit(X_train,t_train)
    return (1 - tree.score(X_test,t_test), (1 - cross_val_score(tree, X_test, t_test, cv=5)).mean())


def plotDecisionTree():
    print("Task 1 assigned to thread")

    best_num, numLeaves, errVals, cvErrs, min_err = getErrors((2,400,1),decisionTree)

    global min_dt_err
    min_dt_err = min_err

    plt.figure()
    plt.plot(numLeaves, errVals, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Decision Tree: Number of Leaf Nodes vs Error")
    plt.xlabel("Number of Leaves")
    plt.ylabel("Error")
    plt.savefig("DecisionTreeErrs.png", format = 'png')

    plt.figure()
    plt.plot(numLeaves, cvErrs, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.title("Decision Tree: Number of Leaf Nodes vs Cross Validation Error")
    plt.xlabel("Number of Leaves")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("DecisionCVTreeErrs.png", format = 'png')

    print(best_num)


#bagging
def baggingErrors(estimators):
    bagging = BaggingClassifier(random_state =RANDOM_SEED, n_estimators=estimators).fit(X_train,t_train)
    return (1 - bagging.score(X_test,t_test), 1 - cross_val_score(bagging, X_test, t_test, cv=5).mean())


def plotBagging():
    print("Task 2 assigned to thread")

    best_num, numEstimators, errVals, cvErrs, _ = getErrors((50,2500,50),baggingErrors)
    
    plt.figure()
    plt.plot(numEstimators, errVals, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Bagging: Number of Estimators vs Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("BaggingErrs.png", format = 'png')

    plt.figure()
    plt.plot(numEstimators, cvErrs, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Bagging: Number of Estimators vs Cross Validation Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("BaggingCVErrs.png", format = 'png')

    print(best_num)

#random forest
def randomForestErrors(estimators):
    rforest = RandomForestClassifier(random_state =RANDOM_SEED, n_estimators=estimators).fit(X_train,t_train)
    return (1 - rforest.score(X_test,t_test), 1 - cross_val_score(rforest, X_test, t_test, cv=5).mean())


def plotRandomForest():
    print("Task 3 assigned to thread")

    best_num, numEstimators, errVals, cvErrs, _ = getErrors((50,2500,50),randomForestErrors)

    plt.figure()
    plt.plot(numEstimators, errVals, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Random Forest: Number of Estimators vs Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("RandomeForestErrs.png", format = 'png')

    plt.figure()
    plt.plot(numEstimators, cvErrs, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Random Forest: Number of Estimators vs Cross Validation Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("RandomeForestCVErrs.png", format = 'png')

    print(best_num)

#Adaboost
def adaboostErrors(estimators):
    adaboost = AdaBoostClassifier(random_state = RANDOM_SEED, n_estimators = estimators, base_estimator = DecisionTreeClassifier(max_depth = 1)).fit(X_train,t_train)
    return (1 - adaboost.score(X_test,t_test), 1 - cross_val_score(adaboost, X_test, t_test, cv=5).mean())


def plotAdaboost():
    print("Task 4 assigned to thread")

    best_num, numEstimators, errVals, cvErrs, _ = getErrors((50,2500,50),adaboostErrors)

    plt.figure()
    plt.plot(numEstimators, errVals, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Adaboost: Number of Estimators vs Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostErrs.png", format = 'png')

    plt.figure()
    plt.plot(numEstimators, cvErrs, label = 'Error of Predictor')
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.title("Adaboost: Number of Estimators vs Cross Validation Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostCVErrs.png", format = 'png')

    print(best_num)

#Adaboost maxLeaves 10
def adaboostErrorsMax10(estimators):
    adaboost = AdaBoostClassifier(random_state = RANDOM_SEED, n_estimators = estimators, base_estimator = DecisionTreeClassifier(max_leaf_nodes = 10)).fit(X_train,t_train)
    return (1 - adaboost.score(X_test,t_test), 1 - cross_val_score(adaboost, X_test, t_test, cv=5).mean())


def plotAdaboostMax10():
    print("Task 5 assigned to thread")

    best_num, numEstimators, errVals, cvErrs, _ = getErrors((50,2500,50),adaboostErrorsMax10)

    plt.figure()
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.plot(numEstimators, errVals, label = 'Error of Predictor')
    plt.title("Adaboost Max Leaves = 10: Number of Estimators vs Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostErrsMax10.png", format = 'png')

    plt.figure()
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.plot(numEstimators, cvErrs, label = 'Error of Predictor')
    plt.title("Adaboost Max Leaves = 10: Number of Estimators vs Cross Validation Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostCVErrsMax10.png", format = 'png')

    print(best_num)

#adaboost no limit
def adaboostErrorsNoLimit(estimators):
    adaboost = AdaBoostClassifier(random_state = RANDOM_SEED, n_estimators = estimators, base_estimator = DecisionTreeClassifier(max_depth=math.inf, max_leaf_nodes=math.inf)).fit(X_train,t_train)
    return (1 - adaboost.score(X_test,t_test), 1 - cross_val_score(adaboost, X_test, t_test, cv=5).mean())


def plotAdaboostNoLimit():
    print("Task 6 assigned to thread")

    best_num, numEstimators, errVals, cvErrs, _ = getErrors((50,2500,50),adaboostErrorsMax10)

    plt.figure()
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.plot(numEstimators, errVals, label = 'Error of Predictor')
    plt.title("Adaboost No Limit: Number of Estimators vs Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostErrsNoLimit.png", format = 'png')

    plt.figure()
    plt.axhline(y = min_dt_err, color = 'red', label = 'Test Error of best DT number of leaves')
    plt.legend()
    plt.plot(numEstimators, cvErrs, label = 'Error of Predictor')
    plt.title("Adaboost No Limit: Number of Estimators vs Cross Validation Error")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error")
    plt.savefig("AdaboostCVErrsNoLimit.png", format = 'png')

    print(best_num)



if __name__ == "__main__":
    #load data
    dataset = pd.read_csv(DATA_FILEPATH)
    X = dataset.iloc[:,:-1].values
    t = dataset.iloc[:,-1].values

    #split dataset
    
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/3, random_state = RANDOM_SEED)
    

    '''Uncomment below for specific functionality'''
    '''For some reason, I wasn't able to force a GIL unlock and the threading executes sequentially'''
    bestDecTreeLeaf = Process(target = plotDecisionTree())
    bestBagging = Process(target = plotBagging())
    bestRandomForest = Process(target = plotRandomForest())
    bestAdaboost = Process(target = plotAdaboost())
    bestAdaboostMax10 = Process(target = plotAdaboostMax10())
    bestAdaboostNoLimit = Process(target = plotAdaboostNoLimit())

    bestDecTreeLeaf.start()
    bestBagging.start()
    bestRandomForest.start()
    bestAdaboost.start()
    bestAdaboostMax10.start()
    bestAdaboostNoLimit.start()

    
    bestDecTreeLeaf.join()
    bestBagging.join()
    bestRandomForest.join()
    bestAdaboost.join()
    bestAdaboostMax10.join()
    bestAdaboostNoLimit.join()