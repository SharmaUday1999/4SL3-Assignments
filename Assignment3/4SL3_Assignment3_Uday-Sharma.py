import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import KFold



#manual logRegression
def logisticRegressionGD():
    new_col=np.ones(N)
    X_train_1 = np.insert(X_train, 0, new_col, axis=1) #add the dummy column
    new_col=np.ones(M)
    X_test_1 = np.insert(X_test, 0, new_col, axis=1) #add the dummy column
    alpha = 0.01
    w = np.array([1 for i in range(31)])# inial w
    y = np.zeros(N)
    y_v = np.zeros(M)
    IT = 2000 # number of iterations of GD
    cost = np.zeros(IT)
    cost_v = np.zeros(IT)

    for n in range(IT):
        z_train = np.dot(X_train_1, w)
        z_test = np.dot(X_test_1, w)
        y = 1/(1 + np.exp(-z_train))
        y_v = 1/(1 + np.exp(-z_test))

        diff = y - t_train
        grad = np.dot(X_train_1.T, diff)/N

        w = w - alpha * grad

        for i in range(N):
            cost[n] += t_train[i] * np.logaddexp(0,-z_train[i]) + (1-t_train[i]) * np.logaddexp(0, z_train[i])
        cost[n] /= N

        for i in range(M):
            cost_v[n] += t_test[i] * np.logaddexp(0,-z_test[i]) + (1-t_test[i]) * np.logaddexp(0, z_test[i])
        cost_v[n] /= M

    temp = np.zeros(M)
    for i in range(M):
        if(z_test[i] >= 0):
            temp[i] = 1
    

    MR = temp - t_test
    err = np.count_nonzero(MR)/M

    precision, recall, threshold = precision_recall_curve(t_test, y_v)
    plt.plot(recall,precision)
    plt.title('Precision/Recall Manual')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    print(f1_score(t_test,temp))
    plt.show()
    return err

#sciKit LogReg Implementation
def logisticRegressionSciKit():
    x = np.reshape(X_train,(len(X_train),-1))
    x_t = np.reshape(X_test,(len(X_test),-1))
    model = LogisticRegression(solver='liblinear', random_state=0).fit(x, t_train)
    y_v = model.predict(x_t)
    plot_precision_recall_curve(model, X_test, t_test)
    plt.title('Precision/Recall SciKit')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(f1_score(t_test,y_v))
    return model.score(x_t, t_test)

#scikit K-fold implementation
def KFoldSciKit(K):
    kf = KFold(n_splits=5)
    errors = []
    for train_index, test_index in kf.split(X):

        curr_X = X[list(train_index),:]
        trainPortion =  t[train_index]
        currX_train = np.reshape(curr_X,(len(curr_X),-1))
        neigh = KNeighborsClassifier(n_neighbors=K).fit(currX_train,trainPortion)
        
        testPortion = t[test_index]
        y = neigh.predict(X[test_index,:])

        diff_test = np.subtract(y,testPortion)
        error_test = np.dot(diff_test.T,diff_test)/len(test_index)
        errors.append(error_test)
    return sum(errors)/5

#scikit KNN Implementation
def kNNSciKit():
    bestK = 1
    minError = math.inf
    errs = []

    for k in range(1,6):
        currError = KFoldSciKit(k)
        errs.append(currError)
        if currError < minError:
            minError = currError
            bestK = k

    
    neigh = KNeighborsClassifier(n_neighbors=bestK).fit(X_train,t_train)
    y = neigh.predict(X_test)
    
    print(errs)
    
    print(f1_score(t_test,y))
    return minError

#manual KNN implmentation with a max-heap
def kFoldManual(K):
    #need to build a heap
    kf = KFold(n_splits=5)
    errors = []
    for train_index, test_index in kf.split(X):

        curr_X = X[list(train_index),:]
        
        #do KNN here manually
        y = [0 for i in range(len(test_index))]

        for i in range(len(test_index)):
            numHits = 0
            for feature in range(30):
                curr_point = X[test_index[i]][feature]
                heap = []

                for each in train_index:
                    dist = abs(curr_point-X[each][feature])
                    if len(heap) == K and dist < -heap[0][0]: #max heap optimization for KlogK heap operations
                        heapq.heappushpop(heap,(-dist,-t[each]))
                    elif len(heap) < K:
                        heapq.heappush(heap,(-dist,-t[each]))
                
                pred = 0
                
                while heap:
                    pred += -heapq.heappop(heap)[1]
                
                if pred/K == 0.5:
                    pred = 1
                else:
                    pred = round(pred/K)
                
                if pred:
                    numHits += 1
            
            y[i] = numHits//16
        
        testPortion = t[test_index]

        diff_test = np.subtract(y,testPortion)
        error_test = np.dot(diff_test.T,diff_test)/len(test_index)
        errors.append(error_test)

    return sum(errors)/5
        
#wrapper function for manual KNN
def kNNManual():
    bestK = 1
    minError = math.inf
    errs = []

    for k in range(1,6):
        currError = kFoldManual(k)
        errs.append(currError)
        if currError < minError:
            minError = currError
            bestK = k

    y = [0 for i in range(len(t_test))]

    for i in range(len(t_test)):
        numHits = 0
        for feature in range(30):
            curr_point = X_test[i][feature]
            heap = []

            for each in range(len(X_train)):
                dist = abs(curr_point-X_train[each][feature])
                if len(heap) == bestK and dist < -heap[0][0]: #max heap optimization for KlogK heap operations
                    heapq.heappushpop(heap,(-dist,-t[each]))
                elif len(heap) < bestK:
                    heapq.heappush(heap,(-dist,-t[each]))
                
            pred = 0
                
            while heap:
                pred += -heapq.heappop(heap)[1]
                
            if pred/bestK == 0.5:
                    pred = 1
            else:
                pred = round(pred/bestK)
                
            if pred:
                numHits += 1
            
        y[i] = numHits//16
    
    print(errs)
    print(f1_score(t_test,y))

    return minError

#main function for variable declaration and function calls
if __name__ == "__main__":
    breast_cancer = load_breast_cancer()
    X, t = load_breast_cancer(return_X_y=True)

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/5, random_state = 9248)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    N = len(X_train)
    M = len(X_test)

    errs = []

    errs.append(1-logisticRegressionSciKit()) #subtract by 1 as the score function is used for this
    errs.append(logisticRegressionGD())
    errs.append(kNNSciKit())
    errs.append(kNNManual())
