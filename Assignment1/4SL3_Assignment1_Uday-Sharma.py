#imports
import numpy as np
import matplotlib.pyplot as plt


def trainForDifferentCapacities():
    global y_for_9
    global predictors_for_9
    for i in range(0,10):
        #handle first iteration
        
        if i == 0:
            X = np.c_[np.ones(N)]   
            X_validation = np.c_[np.ones(M)] 

        elif i == 1:
            X = np.c_[np.ones(N), X_train]   
            X_validation = np.c_[np.ones(M), X_valid] 
        #otherwise we just append the next result to the existing array

        #if M == 9 then I need to use L^2 regularization
        else:
            X = np.c_[X, X_train**i] 
            X_validation = np.c_[X_validation, X_valid**i] 
        
        w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,t_train))
        y = np.dot(X,w)

        if i == 9:
            predictors_for_9 = w
            y_for_9 = y

        y_valid = np.dot(X_validation,w)

        #calculate and store training error
        diff_train = np.subtract(y,t_train)
        err_train = np.dot(diff_train.T,diff_train)/N
        trainingError.append(err_train) #for different M

        #calculate and store each validation error
        diff_valid = np.subtract(y_valid,t_valid)
        err_valid = np.dot(diff_valid.T,diff_valid)/M      
        validationError.append(err_valid)

        #store our predicted values for both the training and validation set
        y_forDiffCapacity.append(y)
        y_validation.append(y_valid)


#implement regularization for M = 9 case:
def performRegularization():
    X_train_for_M_is_9 = np.c_[np.ones(N), X_train, X_train**2, X_train**3, X_train**4, X_train**5, X_train**6, X_train**7, X_train**8, X_train**9] 
    X_valid_for_M_is_9 = np.c_[np.ones(M), X_valid, X_valid**2, X_valid**3, X_valid**4, X_valid**5, X_valid**6, X_valid**7, X_valid**8, X_valid**9]
    I = np.eye((N))
    for item in lambdas:
        w = np.dot(np.linalg.inv(np.dot(X_train_for_M_is_9.T,X_train_for_M_is_9 + item*I)),np.dot(X_train_for_M_is_9.T,t_train)) #determing the parameters for each hyperparameter -> lambda
        y_for_M_is_9_valid.append(np.dot(X_valid_for_M_is_9,w)) #store the result


def plotDifferentLambdas():
    count = 1
    i = 0
    j = 0

    figure, axis = plt.subplots(5, 2) #plot multiple subplots on a single figure

    for result in y_for_M_is_9_valid:
        #plot results
        if i == 5:
            i = 0
            j+=1
        axis[i,j].set_title(str("lambda = " + str(lambdas[count-1])))
        axis[i,j].scatter(X_train, t_train, color = 'magenta', label = 'training example')
        axis[i,j].scatter(X_valid, t_valid, color = 'blue', label = 'test example')
        axis[i,j].scatter(X_valid, f_true, color = 'yellow', label = 'true example')
        axis[i,j].plot(X_valid,result , color = 'green', label = 'prediction')
        #adjust the location of the legend to not block the graph
        box = axis[i,j].get_position()
        axis[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axis[i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i+=1
        count += 1

    plt.show()


def plotDifferentCapacities():
    count = 0
    i = 0
    j = 0

    figure, axis = plt.subplots(5, 2)

    for result in y_validation:
        #plot results
        if i == 5:
            i = 0
            j+=1
        axis[i,j].set_title(str("M = " + str(count)))
        axis[i,j].scatter(X_train, t_train, color = 'magenta', label = 'training example')
        axis[i,j].scatter(X_valid, t_valid, color = 'blue', label = 'test example')
        axis[i,j].scatter(X_valid, f_true, color = 'yellow', label = 'true example')
        axis[i,j].plot(X_valid,result , color = 'green', label = 'prediction')
        #adjust the location of the legend to not block the graph
        box = axis[i,j].get_position() 
        axis[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axis[i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i+=1
        count += 1

    plt.show()


def plotTrainingErrors(): #this plots the training and validation errors vs M
    count = 0
    i = 0
    j = 0

    arr = [0,1,2,3,4,5,6,7,8,9]

    print(np.array(validationError)-np.array(trainingError))

    plt.title("Plot of Training and Validation errors for different Model Capacities")
    plt.scatter(arr,trainingError , color = 'red', label = 'training error')
    plt.plot(arr,trainingError, color = 'red')
    plt.scatter(arr,validationError , color = 'blue', label = 'validation error')
    plt.plot(arr,validationError, color = 'blue')
    plt.scatter(arr,np.array(validationError)-np.array(trainingError) , color = 'green', label = 'difference')
    plt.plot(arr,np.array(validationError)-np.array(trainingError), color = 'green')
    plt.xlabel("M Value")
    plt.ylabel("Error")
    plt.legend()

    plt.show()



if __name__ == "__main__":
    #data generation
    X_train = np.linspace(0.,1.,10)
    X_valid = np.linspace(0.,1.,100)
    f_true = np.sin(4*np.pi*X_valid)
    np.random.seed(9248)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)

    #variable declaration
    y_forDiffCapacity = []
    y_validation = []
    trainingError = []
    validationError = []

    y_for_M_is_9_valid = []

    lambdas = [1e-15, 1e-10, 1e-9, 1e-8, 1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    N = len(X_train)
    M = len(X_valid)

    X = []
    X_validation = []

    trainForDifferentCapacities()
    performRegularization()

    #plot the different plots required
    plotDifferentCapacities()
    plotDifferentLambdas()
    plotTrainingErrors()
