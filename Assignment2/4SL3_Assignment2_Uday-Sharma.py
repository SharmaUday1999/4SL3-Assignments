from os import error
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,train_test_split
import heapq

#start with e

#should only have to do one calulation for each feature
#generate linear regression models

def crossValidation(currData):
    kf = KFold(n_splits=5,random_state=9248,shuffle=True)

    errors = []
    for train_index, test_index in kf.split(currData):
    
        X = np.c_[np.ones(len(list(train_index))),currData[list(train_index), :]]
        trainPortion = t[train_index]

        X_validation = np.c_[np.ones(len(list(test_index))), currData[list(test_index),:]]
        testPortion = t[test_index]
        
        w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,trainPortion))
        y = np.dot(X_validation,w)
        
        diff_test = np.subtract(y,testPortion)
        err_test = np.dot(diff_test.T,diff_test)/len(train_index)
        errors.append(err_test)
    return sum(errors)/5


def linRegBasis():
    X_b = np.concatenate((X_train[:,currFeatures], X_train[:,currFeatures]**2), axis = 1)
    X_validation_b = np.concatenate((X_test[:,currFeatures],X_test[:,currFeatures]**2),axis = 1)

    X_one_b = np.concatenate((X_train[:,currFeatures],X_train[:,currFeatures]**2, X_train[:,currFeatures]**3),axis = 1)
    X_validation_one_b = np.concatenate((X_test[:,currFeatures], X_test[:,currFeatures]**2, X_test[:,currFeatures]**3), axis = 1)

    w = np.dot(np.linalg.inv(np.dot(X_b.T,X_b)),np.dot(X_b.T,t_train))
    w_one = np.dot(np.linalg.inv(np.dot(X_one_b.T,X_one_b)),np.dot(X_one_b.T,t_train))

    y_valid = np.dot(X_validation_b,w)
    y_valid_one = np.dot(X_validation_one_b,w_one)

    diff_valid = np.subtract(y_valid,t_test)
    err_valid = np.dot(diff_valid.T,diff_valid)/M 
    
    diff_valid = np.subtract(y_valid_one,t_test)
    err_valid_one = np.dot(diff_valid.T,diff_valid)/M 

    return (err_valid, err_valid_one, crossValidation(X_b), crossValidation(X_validation_one_b))


def linReg(features):
    X = np.c_[np.ones(N),X_train[:,features]]
    X_validation = np.c_[np.ones(M),X_test[:,features]]

    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,t_train))
    y_valid = np.dot(X_validation,w)

    diff_valid = np.subtract(y_valid,t_test)
    err_valid = np.dot(diff_valid.T,diff_valid)/M 

    return (err_valid,np.dot(X,w),w)


def getLinearRegressionModel(featureIndex): #gets model
    #need to find 
    
    kf = KFold(random_state=9248,shuffle=True)
    errors = []

    for train_index, test_index in kf.split(X_data[:,currFeatures + [featureIndex]]):

        X = np.c_[np.ones(len(list(train_index))),X_data[list(train_index), :][:, currFeatures + [featureIndex]]]
        trainPortion = t[train_index]

        X_validation = np.c_[np.ones(len(list(test_index))), X_data[list(test_index),:][:,currFeatures + [featureIndex]]]
        testPortion = t[test_index]
        
        w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,trainPortion))
        y = np.dot(X_validation,w)
        
        diff_test = np.subtract(y,testPortion)
        err_test = np.dot(diff_test.T,diff_test)/len(train_index)
        errors.append(err_test)

    return (sum(errors)/5,featureIndex)


if __name__ == "__main__":
    boston = load_boston()
    #print(diabetes.DESCR) #run this to see the description of the data set
    X_data, t = load_boston(return_X_y=True)
    X_train, X_test, t_train, t_test = train_test_split(X_data, t, test_size = 0.2, random_state = 9248)

    N = len(X_train)
    M = len(X_test)     

    currFeatures = [] #retains order
    crossValidationErrors = [[] for i in range(13)]


    errorsForModels = {}
    errorsForBasisExpansion = {}
    CVErrorsForBasisExpansion = {}
    CVErrorsForSelectedFeature = {}

    seenFeatures = set()

    #for each feature, build a resultant priority queue, where we pick the smallest one    
    for i in range(13):
        errorsToModels = []

        for j in range(np.shape(X_data)[1]):
            if j not in seenFeatures:
                (error,feature) = getLinearRegressionModel(j)
                crossValidationErrors[i].append(error)
                heapq.heappush(errorsToModels,(error,feature))

        #best model so far


        curr = heapq.heappop(errorsToModels)
        CVErrorsForSelectedFeature[i] = curr[0]
        seenFeatures.add(curr[1])
        currFeatures.append(curr[1])

        #Linear regression on the currFeatures at this point
        errorsForModels[i] = linReg(currFeatures)[0]

        errorsToModels = []

        #basis expansion on the currFeatures at this point

        #FOR SOME REASON THERE ARE SOME SINGULAR MATRICIES THAT EXIST, IDK WHY
        try:
            errBasis1,errBasis2,crossErrorBasis1,crossErrorBasis2 = linRegBasis()
            errorsForBasisExpansion[i] = (errBasis1,errBasis2)
            CVErrorsForBasisExpansion[i] = (crossErrorBasis1,crossErrorBasis2)
        except:
            continue
        #tack on currFeatures squared at the end

        #need to do the same for the validation set


    #now that we have the optimal features to use, we train the data set only using these


    #print out the tables and plots for better viewing of specific info
    print("\n")
    print("\n")
    print("\n")
    print("Linear Regression without Basis Expansion")
    print("_________________________________________")
    print('Number of features | Error')
    for key in errorsForModels:
        print('                {}     {}'.format(key+1, errorsForModels[key]))

    print("\n")
    print("\n")
    print("\n")
    print("Cross validation errors Linear Regression with Basis Expansion")
    print("_________________________________________")
    print('Number of features | Error (Basis Expansion 1, Basis Expansion 2)')
    for key in CVErrorsForBasisExpansion:
        print('                {}     {}'.format(key+1, CVErrorsForBasisExpansion[key]))



    print("\n")
    print("\n")
    print("\n")
    print("Cross Validation errors for determining what features to select and in what order")
    print("_________________________________________")
    print('Number of features So Far | Possible Features | Error')

    for i in range(len(crossValidationErrors)):
        for j in range(len(crossValidationErrors[i])):
            print('                       {}         {}              {}'.format(i+1,j+1, crossValidationErrors[i][j]))


    print("\n")
    print("\n")
    print("\n")
    print(currFeatures)

    y_to_plot = linReg(currFeatures[:7])[1]
    print(linReg(currFeatures[:7])[2])

    
    plt.scatter(X_train[:,0], t_train, color = 'magenta', label = 'Training Examples from Models 1-7')
    plt.scatter(X_train[:,1], t_train, color = 'magenta')
    plt.scatter(X_train[:,2], t_train, color = 'magenta')
    plt.scatter(X_train[:,3], t_train, color = 'magenta')
    plt.scatter(X_train[:,4], t_train, color = 'magenta')
    plt.scatter(X_train[:,5], t_train, color = 'magenta')
    plt.scatter(X_train[:,6], t_train, color = 'magenta')
    plt.scatter(X_train[:,7], t_train, color = 'magenta')
    plt.scatter(X_train[:,0], y_to_plot, color = 'green', label = 'Predicted Values')
    plt.scatter(X_train[:,1], y_to_plot, color = 'green')
    plt.scatter(X_train[:,2], y_to_plot, color = 'green')
    plt.scatter(X_train[:,3], y_to_plot, color = 'green')
    plt.scatter(X_train[:,4], y_to_plot, color = 'green')
    plt.scatter(X_train[:,5], y_to_plot, color = 'green')
    plt.scatter(X_train[:,6], y_to_plot, color = 'green')
    plt.scatter(X_train[:,7], y_to_plot, color = 'green')
    plt.legend()
    plt.show()

    for i in range(12):
        plt.scatter(currFeatures[i]+1, CVErrorsForSelectedFeature[i],  color = 'magenta')
        plt.plot(currFeatures[i]+1, CVErrorsForSelectedFeature[i],  color = 'magenta')
    plt.scatter(currFeatures[12]+1, CVErrorsForSelectedFeature[12], color = 'magenta')
    plt.plot(currFeatures[12]+1, CVErrorsForSelectedFeature[12], color = 'magenta', label = 'Cross Validation Errors')

    for i in range(12):
        plt.scatter(currFeatures[i]+1, errorsForModels[i], color = 'blue', label = '')
        plt.plot(currFeatures[i]+1, errorsForModels[i], color = 'blue', label = '')
    plt.scatter(currFeatures[12]+1, errorsForModels[12], color = 'blue')
    plt.plot(currFeatures[12]+1, errorsForModels[12], color = 'blue', label = 'Test Errors')
    plt.xlabel("Number Of Features")
    plt.ylabel("Error")
    plt.legend()

    plt.show()


    for i in range(5):
        plt.scatter(i+1, errorsForBasisExpansion[i][1],  color = 'magenta')
        plt.scatter(i+1, CVErrorsForBasisExpansion[i][1],  color = 'blue')
    plt.scatter(5+1, errorsForBasisExpansion[5][1],  color = 'magenta', label = 'Test Errors for models with Basis Expansion')
    plt.scatter(5+1, CVErrorsForBasisExpansion[5][1],  color = 'blue', label = 'Cross Validation errors for models with Basis Expansion')
    plt.xlabel("Number Of Features")
    plt.ylabel("Error")
        
    plt.legend()
    plt.show()

